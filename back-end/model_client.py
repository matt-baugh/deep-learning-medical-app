from typing import List
import os

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
    POPULATION_LOC: {  # Test F-1 score 0.9
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
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH_BASE, model_config[PATH])))
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
    pred_prob, pred_ind, att_map = query_model(input_tensor, PATIENT_MODEL)

    # produce an output string to display on front-end
    classes = {0: 'healthy', 1: 'abnormal (Crohn\'s)'}
    output_str = f'{classes[pred_ind]} with probability {round(pred_prob, 3)}'

    att_map_img = make_attention_map_image(patient, att_map)
    sitk.WriteImage(att_map_img, './feature_map_image.nii')

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

    sample_data = torch.stack(tensors).float()

    # Apply inference time data augmentation
    center_crop = _center_crop_gen([OUT_HIGH if even_res else OUT_LOW, OUT_HIGH, OUT_HIGH])
    sample_data = center_crop(sample_data)
    sample_data = normalize_3d(sample_data)

    # Unsqueeze so looks like a batch
    return torch.unsqueeze(sample_data, 0)


def query_model(data: torch.Tensor, model: PytorchResNet3D)\
        -> (float, int, np.ndarray):

    print('Querying model...')
    with torch.no_grad():
        data = data.to(device=DEVICE)

        pred, att_map = model.forward(data, True)

        pred, att_map = torch.squeeze(pred.cpu()), att_map.cpu()

        pred_probs = torch.nn.functional.softmax(pred, dim=0)

        pred_prob, pred_index = torch.max(pred_probs, dim=0)

        # Rescale attention map to be same size as input
        scaled_att_map = torch.nn.functional.interpolate(att_map, data.shape[2:], mode='trilinear', align_corners=True)

    return pred_prob.item(), pred_index.item(), torch.squeeze(scaled_att_map).numpy()


def make_attention_map_image(patient, att_map):
    att_map_img = sitk.GetImageFromArray(att_map)
    out_dim = np.array(att_map_img.GetSize())

    template_img = patient.axial_image
    in_dim = np.array(template_img.GetSize())

    # Axial image still has buffer around center, need to perform centre crop before we copy info
    pixel_origin = (in_dim - out_dim) // 2
    template_img = template_img[pixel_origin[0]: pixel_origin[0] + out_dim[0],
                                pixel_origin[1]: pixel_origin[1] + out_dim[1],
                                pixel_origin[2]: pixel_origin[2] + out_dim[2]]

    att_map_img.CopyInformation(template_img)
    return att_map_img


if __name__ == "__main__":
    test_coords = [281, 258, 44]
    get_patient_prediction(test_coords, "../examples/A1 Axial T2.nii", "../examples/A1 Coronal T2.nii",
                           "../examples/A1 Axial Postcon.nii")
