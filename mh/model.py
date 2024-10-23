import logging
from io import BytesIO
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, DATA_UNDEFINED_NAME

logger = logging.getLogger(__name__)

class NewModel(LabelStudioMLBase):
    def __init__(self, project_id, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        logger.info('Initializing NewModel')
        parsed_config = self.get('parsed_label_config')
        logger.debug(f'Parsed label config: {parsed_config}')
        
        # Store API configurations
        self.label_studio_auth = 'Token bd09a8fe874cfaf2e7a884dca7cf431fac06e6e4'
        self.label_studio_base_url = 'https://labelstudio.server01.mhcam.app'
        self.ml_server_url = 'https://ml.server01.mhcam.app/detector'

    def _get_image_dimensions(self, image_url):
        """Helper method to get image dimensions safely"""
        try:
            response = requests.get(
                image_url, 
                headers={'Authorization': self.label_studio_auth},
                timeout=10
            )
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img.size
        except Exception as e:
            logger.error(f'Error getting image dimensions: {str(e)}')
            raise

    def predict(self, tasks, context=None, **kwargs):
        """Make predictions for the given tasks"""
        if not tasks:
            logger.warning('No tasks provided for prediction')
            return [{"results": []}]

        try:
            # Get image URL and dimensions
            task = tasks[0]
            image_path = task.get('data', {}).get('image')
            if not image_path:
                logger.error('No image path found in task data')
                return [{"results": []}]
                
            image_url = f'{self.label_studio_base_url}{image_path}'
            logger.info(f'Processing image: {image_url}')
            
            # Get image dimensions
            img_width, img_height = self._get_image_dimensions(image_url)
            logger.debug(f'Image dimensions: {img_width}x{img_height}')
            
            # Make prediction request
            logger.debug(f'Sending request to ML server: {self.ml_server_url}')
            response = requests.post(
                self.ml_server_url,
                json={'img': image_url},
                timeout=30
            )
            
            # Log the raw response for debugging
            logger.debug(f'ML server status code: {response.status_code}')
            logger.debug(f'ML server response headers: {response.headers}')
            logger.debug(f'ML server raw response: {response.text}')
            
            # Handle non-200 responses
            if response.status_code != 200:
                logger.error(f'ML server returned status code {response.status_code}')
                return [{"results": []}]
            
            # Try to parse the response
            try:
                detection_result = response.json()
                logger.debug(f'Parsed detection result: {detection_result}')
                
                # If we get an empty result or error message
                if not detection_result or 'error' in detection_result:
                    logger.warning(f'Empty or error result from ML server: {detection_result}')
                    return [{"results": []}]
                
                # Process valid results
                results = []
                all_scores = []
                
                for item in detection_result:
                    bbox = item.get("bboxPercent", [])
                    if len(bbox) < 4:
                        logger.warning(f'Invalid bbox format: {bbox}')
                        continue
                        
                    x, y, xmax, ymax = bbox[:4]
                    score = item.get("score", 0)
                    output_label = item.get("class", "unknown")
                    
                    results.append({
                        'from_name': 'label',
                        'to_name': 'image',
                        'type': 'rectanglelabels',
                        'original_width': img_width,
                        'original_height': img_height,
                        'image_rotation': 0,
                        'value': {
                            'rotation': 0,
                            'rectanglelabels': [output_label],
                            'x': x,
                            'y': y,
                            'width': xmax,
                            'height': ymax
                        },
                        'score': score
                    })
                    all_scores.append(score)
                
                if results:
                    avg_score = sum(all_scores) / len(all_scores)
                    return [{
                        'result': results,
                        'score': avg_score
                    }]
                else:
                    return [{"results": []}]
                    
            except requests.exceptions.JSONDecodeError as e:
                logger.error(f'Failed to decode JSON response: {str(e)}')
                logger.error(f'Raw response text: {response.text}')
                return [{"results": []}]
                
        except requests.exceptions.RequestException as e:
            logger.error(f'Request error: {str(e)}')
            return [{"results": []}]
        except Exception as e:
            logger.error(f'Unexpected error in predict: {str(e)}')
            return [{"results": []}]

    def fit(self, event, data, **kwargs):
        """Handle annotation events"""
        logger.debug(f'Fit called with event: {event}')
        pass