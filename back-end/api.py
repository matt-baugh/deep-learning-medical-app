import os
import glob
from flask import Flask, request, session, send_file, make_response
from flask_cors import CORS, cross_origin
from model_client import get_patient_prediction
import sys

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['nii', 'nii.gz'])

AXIALT2 = 'axialT2'
CORONALT2 = 'coronalT2'
AXIALPC = 'axialPC'

SCAN_TYPES = [AXIALT2, CORONALT2, AXIALPC]

app = Flask(__name__)

cors = CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/test', methods=['GET'])
def test():
    return 'test works'


@app.route('/upload', methods=['POST'])
def fileUpload():

    scan_type = request.form['scanType']
    if scan_type not in SCAN_TYPES:
        return f'Invalid scan type: {scan_type}', 400

    target = os.path.join(UPLOAD_FOLDER, 'images', scan_type)
    f = request.files['file']
    filename = f.filename
    destination = "/".join([os.path.abspath(os.getcwd()), target, filename])
    f.save(destination)
    #session['uploadFilePath']=destination
    response = "OK"
    return response


@app.route('/predict', methods=['GET'])
def prediction():

    def get_latest_file(folder_name: str):
        target = os.path.join(UPLOAD_FOLDER, 'images', folder_name)
        destination = "/".join([os.path.abspath(os.getcwd()), target, '*'])
        list_of_files = glob.glob(destination)
        return max(list_of_files, key=os.path.getctime)

    axial_t2_path = get_latest_file(AXIALT2)
    coronal_t2_path = get_latest_file(CORONALT2)
    axial_pc_path = get_latest_file(AXIALPC)

    data = request.args
    coords = [int(data.get('x')), int(data.get('y')), int(data.get('z'))]
    showMaps = data.get('showMaps')

    try:
        host_ip = 'crohns'
        print(host_ip, file=sys.stderr)
        pred_str = get_patient_prediction(coords, axial_t2_path, coronal_t2_path, axial_pc_path)
        response = send_file('./feature_map_image.nii',
                             attachment_filename='feature_map_image.nii') if showMaps.lower() == 'true' else make_response()
        response.headers['Score'] = pred_str
        response.headers['Access-Control-Expose-Headers'] = 'Score'
        print(response.headers)

        return response
    except Exception as e:
        print("An error occured when extracting the feature maps.")
        print(e)
        return "An error occured when extracting the feature maps."


app.run(host='0.0.0.0', port=9000, debug=True)
