import React, { useState } from 'react';
import LoadingOverlay from 'react-loading-overlay';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';
import Pagination from 'react-bootstrap/Pagination'
import DragNDrop  from './dragndrop';
import Score from './score';
import Infos from './infos';
import Box from './box';
import upload from '../images/upload_white.svg';
import {axialT2, coronalT2, axialPC, scanTypeNames} from "../constants/frontend";


/**
 * Main component for displaying the center component of the app.
 * Wraps the Infos, DragNDrop, Score component and the papaya viewer
 */
const Visualization = () => {
    const [uploading, setUploading] = useState(false); // whether an image is being

    const [axialT2Uploaded, setAxialT2Uploaded] = useState(false); // whether an axial t2 image has been uploaded
    const [coronalT2Uploaded, setCoronalT2Uploaded] = useState(false); // whether an coronal t2 image has been uploaded
    const [axialPCUploaded, setAxialPCUploaded] = useState(false); // whether an axial pc image has been uploaded

    const [axialT2Scan, setAxialT2Scan] = useState(undefined)
    const [coronalT2Scan, setCoronalT2Scan] = useState(undefined)
    const [axialPCScan, setAxialPCScan] = useState(undefined)

    const [visibleScan, setVisibleScan] = useState(undefined);

    const [error, setError] = useState(null); // if any error occurs during download/upload

    const allUploaded = axialT2Uploaded && coronalT2Uploaded && axialPCUploaded
    const anyUploaded = axialT2Uploaded || coronalT2Uploaded || axialPCUploaded

    const uploadingCallbackGen = (scanType) => {
        /**
         * Triggers uploading process to the back-end
         * If a file has been selected, the papaya viewer is reset and the selected
         * image is loaded into it.
         * After 2 seconds, the coordinate system is changed to physical (matches with
         * model specifications).
         * @param {bool} uploading
         * @param {File[]} files
         */
        return (uploading, files) => {
            setUploading(uploading);
            if (files) {
                window.papaya.Container.resetViewer(0);
                window.papayaContainers[0].viewer.loadBaseImage(files);
                setVisibleScan(scanType);
                switch (scanType) {
                    case axialT2:
                        setAxialT2Scan(files);
                        break;
                    case coronalT2:
                        setCoronalT2Scan(files);
                        break;
                    case axialPC:
                        setAxialPCScan(files)
                        break;
                    default:
                        setError('Invalid scan type: ' + scanType)
                }
            }
        }
    }

    /**
     * Creates a Blob from the heat maps data received from the back-end (as string)
     * and load them into the viewer (stacks them on top of the loaded image)
     * @param {string} data
     */
    const dowloadingFeatureMapsCallback = (data) => {
        var blob = new Blob([data]);
        blob.lastModifiedDate = new Date();
        blob.name = "feature-maps";
        window.papayaContainers[0].toolbar.doAction("OpenImage", [blob], true);
    }

    const errorCallback = () => setError('An error occured. Please try again later.');

    const makeImageSelect = (scanType, scanUploaded, scanFile) => {

        const selectImage = () => {
            window.papayaContainers[0].viewer.loadBaseImage(scanFile);
            setVisibleScan(scanType);
        }

        return <Pagination.Item disabled={!scanUploaded} active={visibleScan === scanType} onClick={selectImage}>
            {scanTypeNames[scanType]} Image
        </Pagination.Item>
    }

    return (
    <Container id="visu" className="py-5 px-5 h-90" style={{ maxWidth: "100%" }}>
        {error ? <Alert variant="danger">{error}</Alert> : null}
        <Row className="row-flex">
            <Col sm={8} className="h-100">
                <Infos />
                <Pagination>
                    {makeImageSelect(axialT2, axialT2Uploaded, axialT2Scan)}
                    {makeImageSelect(coronalT2, coronalT2Uploaded, coronalT2Scan)}
                    {makeImageSelect(axialPC, axialPCUploaded, axialPCScan)}
                </Pagination>
                <Box color="black" title="Papaya Viewer" >
                    <LoadingOverlay
                        active={!anyUploaded && !uploading}
                        spinner={false}
                        text={
                            <div className="d-flex flex-column align-items-center" data-testid='papaya_window' >
                                <img src={upload} alt="upload" height="50px" width="60px" />
                                <span>Please upload an image first</span>
                            </div>
                        }
                        >
                            <div className="papaya" data-params="params"></div>{/* data-params="params" */}
                    </LoadingOverlay>
                </Box>
            </Col>
            <Col sm={4} className="h-100">
                <Box color="white" title="Upload image">
                    <DragNDrop uploadedCallback={setAxialT2Uploaded} uploadingCallback={uploadingCallbackGen(axialT2)} errorCallback={errorCallback} scanType={axialT2}/>
                    <DragNDrop uploadedCallback={setCoronalT2Uploaded} uploadingCallback={uploadingCallbackGen(coronalT2)} errorCallback={errorCallback} scanType={coronalT2}/>
                    <DragNDrop uploadedCallback={setAxialPCUploaded} uploadingCallback={uploadingCallbackGen(axialPC)} errorCallback={errorCallback} scanType={axialPC}/>
                </Box>
                <Score uploaded={allUploaded} uploading={uploading} callback={dowloadingFeatureMapsCallback} patientLoc={true}/>
                <Score uploaded={allUploaded} uploading={uploading} callback={dowloadingFeatureMapsCallback} patientLoc={false}/>
            </Col>
        </Row>
    </Container>);
};

export default Visualization;
