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

    def _process_detection_results(self, detection_result, img_width, img_height):
        """Process detection results into Label Studio format"""
        results = []
        all_scores = []
        
        try:
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
                
            avg_score = sum(all_scores) / max(len(all_scores), 1)
            return [{
                'result': results,
                'score': avg_score
            }]
        except Exception as e:
            logger.error(f'Error processing detection results: {str(e)}')
            raise

    def predict(self, tasks, context=None, **kwargs):
        """Make predictions for the given tasks"""
        if not tasks:
            logger.warning('No tasks provided for prediction')
            return []

        try:
            # Get image URL and dimensions
            task = tasks[0]
            image_path = task.get('data', {}).get('image')
            if not image_path:
                logger.error('No image path found in task data')
                return []
                
            image_url = f'{self.label_studio_base_url}{image_path}'
            logger.debug(f'Processing image: {image_url}')
            
            # Get image dimensions
            img_width, img_height = self._get_image_dimensions(image_url)
            
            # Make prediction request
            response = requests.post(
                self.ml_server_url,
                json={'img': image_url},
                timeout=30
            )
            response.raise_for_status()
            
            # Check for empty response
            if not response.content:
                logger.warning('Empty response from ML server')
                return []
                
            # Parse response
            try:
                detection_result = response.json()
                logger.debug(f'Detection result: {detection_result}')
            except requests.exceptions.JSONDecodeError as e:
                logger.error(f'Failed to decode JSON response: {str(e)}')
                logger.error(f'Raw response: {response.text}')
                return []
                
            # Process results
            return self._process_detection_results(detection_result, img_width, img_height)
            
        except requests.exceptions.RequestException as e:
            logger.error(f'Request error: {str(e)}')
            return []
        except Exception as e:
            logger.error(f'Unexpected error in predict: {str(e)}')
            return []

    def fit(self, event, data, **kwargs):
        """Handle annotation events"""
        logger.debug(f'Fit called with event: {event}')
        # Add training logic here if needed
        pass