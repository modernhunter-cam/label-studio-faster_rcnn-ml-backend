import logging
from io import BytesIO
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

class NewModel(LabelStudioMLBase):
    def __init__(self, project_id, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        logger.info('Initializing NewModel')
        self.label_studio_auth = 'Token bd09a8fe874cfaf2e7a884dca7cf431fac06e6e4'
        self.label_studio_base_url = 'https://labelstudio.server01.mhcam.app'
        self.ml_server_url = 'https://ml.server01.mhcam.app/detector/upload/6'

    def _get_image_file(self, image_url):
        """Download image and return both file data and dimensions"""
        try:
            response = requests.get(
                image_url,
                headers={'Authorization': self.label_studio_auth},
                timeout=10
            )
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return response.content, img.size
        except Exception as e:
            logger.error(f'Error getting image file: {str(e)}')
            raise

    def _make_ml_request(self, image_data):
        """Make request to ML server with image file"""
        try:
            files = {
                'file': ('image.jpg', image_data, 'image/jpeg')
            }
            
            logger.debug('Sending image file to ML server')
            response = requests.post(
                self.ml_server_url,
                files=files,
                timeout=30
            )
            
            logger.debug(f'ML server status: {response.status_code}')
            logger.debug(f'ML server headers: {dict(response.headers)}')
            logger.debug(f'ML server response: {response.text}')
            
            # Accept both 200 and 201 status codes
            if response.status_code not in [200, 201]:
                logger.error(f'ML server error: {response.text}')
                return None
                
            # Check if response is error message
            if response.text.strip() == "Error occurred during prediction":
                logger.error('ML server reported prediction error')
                return None
                
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                logger.error(f'Failed to decode JSON response: {response.text}')
                return None
                
        except Exception as e:
            logger.error(f'Error in ML request: {str(e)}')
            return None

    def predict(self, tasks, context=None, **kwargs):
        """Make predictions for the given tasks"""
        try:
            if not tasks:
                return [{"results": []}]

            task = tasks[0]
            image_path = task.get('data', {}).get('image')
            if not image_path:
                logger.error('No image path in task data')
                return [{"results": []}]
                
            image_url = f'{self.label_studio_base_url}{image_path}'
            logger.info(f'Processing image: {image_url}')
            
            image_data, (img_width, img_height) = self._get_image_file(image_url)
            
            detection_result = self._make_ml_request(image_data)
            if not detection_result:
                return [{"results": []}]
            
            # Process results
            results = []
            all_scores = []
            
            for item in detection_result:
                try:
                    # Use bboxPercent values directly
                    bbox_percent = item.get("bboxPercent", [])
                    if len(bbox_percent) < 4:
                        continue
                        
                    x, y, width, height = bbox_percent
                    score = item.get("score", 0)
                    output_label = item.get("class", "unknown")
                    
                    result = {
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
                            'width': width - x,  # Convert to width
                            'height': height - y  # Convert to height
                        },
                        'score': score / 100  # Convert score to 0-1 range
                    }
                    results.append(result)
                    all_scores.append(score / 100)
                except Exception as e:
                    logger.error(f'Error processing detection item: {str(e)}')
                    continue
            
            if results:
                avg_score = sum(all_scores) / len(all_scores)
                return [{
                    'result': results,
                    'score': avg_score
                }]
            
            return [{"results": []}]
            
        except Exception as e:
            logger.error(f'Unexpected error in predict: {str(e)}')
            return [{"results": []}]

    def fit(self, event, data, **kwargs):
        pass