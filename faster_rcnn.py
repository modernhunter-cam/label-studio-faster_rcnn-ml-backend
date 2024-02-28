import os
import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size, get_single_tag_keys, is_skipped

logger = logging.getLogger(__name__)
feature_extractor_model = 'https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1'


class TFFasterRCNN(LabelStudioMLBase):

    def __init__(self, trainable=False, batch_size=32, epochs=3, confidence_threshold=0.5, **kwargs):
        super(TFFasterRCNN, self).__init__(**kwargs)

        self.trainable = trainable
        self.batch_size = batch_size
        self.epochs = epochs
        self.confidence_threshold = confidence_threshold

        self.feature_extractor = hub.load(feature_extractor_model)

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.num_classes = len(self.labels_in_config)

        self.model = tf.keras.Sequential([
            self.feature_extractor,
            tf.keras.layers.Dense(self.num_classes + 1, activation='softmax')  # +1 for background class
        ])
        self.model.summary()
        if self.train_output:
            model_file = self.train_output['model_file']
            logger.info('Restore model from ' + model_file)
            self.model.load_weights(model_file)

    def predict(self, tasks, **kwargs):
        image_path = get_image_local_path(tasks[0]['data'][self.value])
        image = tf.keras.preprocessing.image.load_img(image_path)
        image = tf.keras.preprocessing.image.img_to_array(image)

        predictions = self.model.predict(image[np.newaxis, ...])
        filtered_predictions = self.filter_predictions(predictions)

        return [{
            'result': [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [{
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1],
                        'label': self.labels_in_config[class_idx]
                    } for bbox, class_idx in filtered_predictions]
                }
            }]
        }]

    def filter_predictions(self, predictions):
        filtered_predictions = []
        for i in range(1, self.num_classes + 1):
            class_predictions = predictions[..., i]
            mask = class_predictions > self.confidence_threshold
            nonzero_indices = tf.where(mask)

            for idx in nonzero_indices:
                confidence = class_predictions[idx[0], idx[1]]
                bbox = idx[1]
                class_idx = i - 1
                filtered_predictions.append((bbox, class_idx, confidence))

        return filtered_predictions

    def fit(self, completions, workdir=None, **kwargs):
        annotations = []
        for completion in completions:
            if is_skipped(completion):
                continue
            image_path = get_image_local_path(completion['data'][self.value])
            image_size = get_image_size(image_path)
            for result in completion['result']:
                for bbox in result['value']['rectanglelabels']:
                    x, y = bbox['x'] / image_size[0], bbox['y'] / image_size[1]
                    width, height = bbox['width'] / image_size[0], bbox['height'] / image_size[1]
                    label = self.labels_in_config.index(bbox['label'])
                    annotations.append((image_path, (x, y, width, height), label))

        ds = tf.data.Dataset.from_tensor_slices(annotations)

        def prepare_item(item):
            img = tf.io.read_file(item[0])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            return img, item[1], item[2]

        ds = ds.map(prepare_item, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])
        self.model.fit(ds, epochs=self.epochs)
        model_file = os.path.join(workdir, 'checkpoint')
        self.model.save_weights(model_file)
        return {'model_file': model_file}


# Example usage:
# classifier = TFFasterRCNN(trainable=True, batch_size=32, epochs=5, confidence_threshold=0.5)
# classifier.fit(completions)
# predictions = classifier.predict(tasks)
