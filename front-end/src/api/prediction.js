import axios from 'axios';

import { SERVER_BASE_URL } from '../constants/api';

export const getPrediction = (patientLoc, x, y, z, onDownloadProgress, showMaps) => axios.get(SERVER_BASE_URL + 'predict?patientLoc=' + (patientLoc ? 1 : 0) + '&x=' + x + '&y=' + y + '&z=' + z + '&showMaps=' + showMaps, {onDownloadProgress, responseType: 'blob'}, {
	headers: {
	  'Access-Control-Allow-Origin': '*',
	},
});
