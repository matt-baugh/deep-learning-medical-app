from typing import List

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import torch
import torchvision.transforms as T

from constants import PATIENT_LOC, POPULATION_LOC, ALL_MODALITIES, EVEN_RES, ATTENTION, PATH
from models.pytorch_resnet import PytorchResNet3D
from preprocessing.metadata import Patient
from preprocessing.preprocess import Preprocessor

MODEL_PATH_BASE = '/vol/bitbucket/mb4617/CrohnsDisease/trained_models'

MODEL_CONFIGS = {
    PATIENT_LOC: {  # Test F-1 Score 0.98, only misclassified 1 healthy sample as unhealthy
        ALL_MODALITIES: True,
        EVEN_RES: False,
        ATTENTION: True,
        PATH: '4/axial_only_extended_dataset_mode1loc1att1/fold0'
    },
    POPULATION_LOC: {
        ALL_MODALITIES: False,
        EVEN_RES: True,
        ATTENTION: True,
        PATH: '6/even_res_extended_dataset_mode0loc0att1/fold1'
    }
}

# Constants for dataset dimensions
# Stored data dimensions
IN_HIGH = 99
IN_LOW = 37
# Model input dimensions
OUT_HIGH = 87
OUT_LOW = 31

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def construct_model(tag):
    model_config = MODEL_CONFIGS[tag]
    model = PytorchResNet3D([OUT_HIGH if model_config[EVEN_RES] else OUT_LOW, OUT_HIGH, OUT_HIGH],
                            model_config[ATTENTION],
                            0.5,
                            3 if model_config[ALL_MODALITIES] else 1)
    model.load_state_dict(torch.load(model_config[PATH]))
    model.eval()
    model.to(DEVICE)
    return model


PATIENT_MODEL = construct_model(PATIENT_LOC)
POPULATION_MODEL = construct_model(POPULATION_LOC)


# Inference time data augmentation
def _center_crop_gen(out_dims):
    final_crop = T.CenterCrop((out_dims[1], out_dims[2]))

    def _center_crop_helper(x):
        dim_0_diff = x.shape[1] - out_dims[0]
        dim_0_crop = dim_0_diff // 2
        return final_crop(x[:, dim_0_crop:dim_0_crop + out_dims[0], :, :])

    return _center_crop_helper


normalize_3d = T.Lambda(
    lambda x: (x - torch.mean(x, dim=[1, 2, 3], keepdim=True)) / torch.std(x, dim=[1, 2, 3], keepdim=True)
)


def get_patient_prediction(coords: List[int], path_to_axialt2: str, path_to_coronalt2: str, path_to_axialpc: str):

    model_config = MODEL_CONFIGS[PATIENT_LOC]

    patient = construct_patient(path_to_axialt2, path_to_coronalt2, path_to_axialpc, model_config[ALL_MODALITIES])

    # Match coordinate system
    # coords [x,y,z] -> [y,x,z] (
    coords = change_coordinate_system(coords, patient)
    patient.set_ileum_coordinates(coords)

    # pre-process image so that it matches input of model
    input_tensor = extract_model_input(patient, model_config[EVEN_RES], model_config[ALL_MODALITIES])

    # query tensorflow seriving model for predictions and attention layer
    prob_values, max_prob_indx, attentions = np.array([[0.5, 0.5]]), 0, None  # query_client(processed_image, client)

    ## process the feature map to get the average and resize it
    # feature_maps_arr = process_feature_maps(attentions, processed_image[0].shape)
    ## make the attention layer into a nifit file
    # make_feature_image(coords, path_to_img, feature_maps_arr)

    # produce an output string to display on front-end
    classes = {0: 'healthy', 1: 'abnormal (Crohn\'s)'}
    predictions = classes[max_prob_indx]
    output_str = f'{predictions} with probability {round(prob_values[0][max_prob_indx], 3)}'

    return output_str


def construct_patient(path_to_axialt2: str, path_to_coronalt2: str, path_to_axialpc: str, all_modalities: bool)\
        -> Patient:
    patient = Patient('UNK', 0)
    patient.set_paths(path_to_axialt2, path_to_coronalt2, path_to_axialpc)
    patient.load_image_data(True, all_modalities, all_modalities)
    return patient


def change_coordinate_system(coords: List[int], patient: Patient) -> List[int]:
    print('initial coordinates: ', coords)

    # load original image and convert to numpy arr
    arr_fig_shape = sitk.GetArrayFromImage(patient.axial_image).shape

    # account for papaya's weird system of changing coordinates
    new_x = arr_fig_shape[2] - coords[0]
    new_y = coords[1]
    new_z = arr_fig_shape[0] - coords[2]

    print('New coordinates: ', new_x, new_y, new_z)

    return [new_y, new_x, new_z]


def extract_model_input(patient: Patient, even_res: bool, all_modalities: bool) -> torch.Tensor:

    # In [x,y,z] to match SITK
    preprop_shape = [IN_HIGH, IN_HIGH, IN_HIGH if even_res else IN_LOW]

    # [z,y,x] -> [y,x,z] but y=x
    preprocessor = Preprocessor(constant_volume_size=preprop_shape)
    [patient] = preprocessor.process([patient], ileum_crop=True, region_grow_crop=False, statistical_region_crop=False)

    images = [patient.axial_image]
    if all_modalities:
        assert patient.coronal_image is not None, 'Model requires all modalities but coronal T2 missing'
        assert patient.axial_postcon_image is not None, 'Model requires all modalities but axial post contrast missing'

        images.append(patient.coronal_image)
        images.append(patient.axial_postcon_image)

    tensors = [torch.from_numpy(sitk.GetArrayFromImage(img)) for img in images]

    sample_data = torch.stack(tensors)

    # Apply inference time data augmentation
    center_crop = _center_crop_gen([OUT_HIGH if even_res else OUT_LOW, OUT_HIGH, OUT_HIGH])
    sample_data = center_crop(sample_data)
    sample_data = normalize_3d(sample_data)

    # Unsqueeze so looks like a batch
    return torch.unsqueeze(sample_data, 0)


def query_model(data: torch.Tensor, model: PytorchResNet3D):

    data = data.to(device=DEVICE)

    pred, att_map = model.forward(data, True)

    pred, att_map = pred.cpu(), att_map.cpu()

    # TODO: CONTINUE
    # query the model with the given data
    out_model = client.predict(req_data)
    prob_values = out_model['Output']
    max_prob_indx = np.argmax(np.squeeze(prob_values))
    # get the attention layers
    attention_layer = out_model['attention_layer']

    return prob_values, max_prob_indx, attention_layer


def process_feature_maps(attention_layer, processed_image_shape):
    # Get the mean of the feature maps
    attention_layer = attention_layer.mean(4)
    # Upsample the attention layer to 87, 87 size
    ratio = tuple(map(lambda x, y: x / y, processed_image_shape, attention_layer.shape))
    upsampled_attention_layer = zoom(attention_layer, ratio)

    return upsampled_attention_layer


def make_feature_image(coords, path, feature_maps_arr):
    # load original image and convert to numpy arr
    loaded_image = sitk.ReadImage(path)
    arr_fig = sitk.GetArrayFromImage(loaded_image).astype("float32")
    # add the maps
    new_arr = add_feature_arra_zero_arr(arr_fig, feature_maps_arr, coords, feature_shape)
    # make it into a nifit file with the same meta-data as original image
    make_arr_into_nifit_image(loaded_image, new_arr)


def add_feature_arra_zero_arr(arr_image, arr_feature_map, pixel_center, physical_crop_size):
    # compute box size
    box_size = np.array([physical_crop_size[1], physical_crop_size[2], physical_crop_size[
        0]])  # np.array([pcsz / vsz for vsz,pcsz in zip(image.GetSpacing(), physical_crop_size)])
    lb = np.array(pixel_center - box_size / 2).astype(int)  # lower corner of cropped box
    ub = (lb + box_size).astype(int)  # upper corner of cropped box
    # fully convert lower bound to Python (=!numpy) format, s.t. it can be used by SITK
    lb = list(lb)
    lb = [int(lower_b) for lower_b in lb]

    # noramlise feature array and fill original image zeros
    arr_feature_map = (arr_feature_map - arr_feature_map.min()) / (arr_feature_map.max() - arr_feature_map.min())
    arr_image = np.zeros(arr_image.shape)

    # get data of cropped box region
    arr_image[lb[2]:ub[2], lb[0]:ub[0], lb[1]:ub[1]] = arr_feature_map  # place the feature map at the given location

    return arr_image.astype(np.float32)


def make_arr_into_nifit_image(base_image, new_image_arr):
    # make the new image array a sitk Image
    feature_map_image = sitk.GetImageFromArray(new_image_arr)
    feature_map_image.CopyInformation(base_image)
    # write to file
    sitk.WriteImage(feature_map_image, './feature_map_image.nii')




if __name__ == "__main__":
    test_coords = [281, 258, 44]
    get_patient_prediction(test_coords, "../examples/A1 Axial T2.nii")
