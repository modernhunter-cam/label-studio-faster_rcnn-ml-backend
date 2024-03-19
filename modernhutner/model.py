import logging
from io import BytesIO

import requests
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME

logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        print(f'__init__ My Class')
        test = self.get('parsed_label_config')
        print(f'parsed_label_config {test}')

    def predict(self, tasks, context=None, **kwargs):
        task = tasks[0]
        image_url = 'https://studio.mhcam.cloud' + task['data']['image']
        img_width, img_height = Image.open(BytesIO(requests.get(image_url, headers={
          'Authorization': 'Token 85e68543b8741a1b6b21aaf449ef3a009f215897'
        }).content)).size
        body = {'img': image_url}
        print(f'predict My Class {body}')
        data = requests.post('https://seahorse.mhcam.cloud/detector', json=body)
        result = data.json()
        # [{'bbox': [37.40971565246582, 427.66900062561035, 300.0874614715576, 422.4660015106201], 'class': 'horse', 'score': 0.9951158761978149}]
        print(f'predict My Class {data.json()}')
        results = []
        all_scores = []
        for item in result:
            bbox = item["bboxPercent"]
            output_label = item["class"]

            x, y, xmax, ymax = bbox[:4]
            score = item["score"]

            results.append({
                'from_name': 'label',  # Adjust if needed
                'to_name': 'image',  # Adjust if needed
                'type': 'rectanglelabels',
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'rotation': 0,  # Adjust if needed
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
        final_result = [{
            'result': results,
            'score': avg_score
        }]
        print(f'predict My Class {final_result}')
        return final_result

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # Add your fit logic here
        pass
