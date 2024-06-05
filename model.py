import os
import torch
from dotenv import load_dotenv
from PIL import Image

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys
from label_studio.core.utils.io import get_data_dir
import label_studio_sdk


load_dotenv()
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

WEIGHTS = './yolov7/yolov7.pt'
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
REPO = "./yolov7"
IMAGE_SIZE = (640, 640)


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def __init__(self, device=DEVICE, img_size=IMAGE_SIZE, repo=REPO, train_output=None, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')

        self.hostname = LABEL_STUDIO_HOST
        self.access_token = LABEL_STUDIO_API_KEY
        self.device = device
        self.img_size = img_size
        self.repo = repo
        self.image_dir = upload_dir

        self.weights = WEIGHTS

        print("------------------Loading the Model---------------")
        self.model = torch.hub.load(
            self.repo, 'custom', self.weights, source='local', trust_repo=True)
        
        print("------------------Model Loaded---------------")


        # print(self.model, "___________model____________")

        # getting all neccessary variable to return from predict method.
        # rectanglelabels
        self.annotation_type = self.parsed_label_config['label']['type']
        # Image
        self.object_type = self.parsed_label_config['label']['inputs'][0]['type']

        self.from_name, self.to_name, self.value, self.annotation_labels = get_single_tag_keys(
            self.parsed_label_config, self.annotation_type, self.object_type
        )

        print(self.from_name, self.to_name, self.value,
              self.annotation_labels, "#####################", self.annotation_type)

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", "0.0.1")

    def download_labeled_tasks(self, project_id):
        """
        Download all labeled tasks from a particular project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/

        Parameters:
            project_id (int): Used to uniquely identify the project we are working.

        Returns:
            list of dict [{}]: returns all labeled tasks.
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)  # For testing keep id=3
        tasks = project.get_labeled_tasks()

        return tasks

    def _get_image_url(self, task):
        # image_url = '/data/upload/4/53c37ced-football_image.jpeg'
        image_url = task['data'][self.value]
        return image_url

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: gets the task of not predicted task which was opened.[Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        ------------------------------------------------
        Received context: {context}
        ------------------------------------------------
        Project ID: {self.project_id}
        ------------------------------------------------
        Label config: {self.label_config}
        ------------------------------------------------
        Parsed JSON Label config: {self.parsed_label_config}
        ------------------------------------------------
        Extra params: {self.extra_params}''')

        print(f"the model uses: {self.weights} to predict")

        results = []
        all_scores = []

        for index, task in enumerate(tasks):
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(
                image_url, project_dir=self.image_dir)
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)

            preds = self.model(img)
            preds_df = preds.pandas().xyxy[0]

            print(preds_df.head())
            for x_min, y_min, x_max, y_max, confidence, class_, name_ in zip(preds_df['xmin'], preds_df['ymin'],
                                                                             preds_df['xmax'], preds_df['ymax'],
                                                                             preds_df['confidence'], preds_df['class'],
                                                                             preds_df['name']):
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    'type': self.annotation_type.lower(),
                    'value': {
                        self.annotation_type.lower(): [name_],
                        'x': x_min / img_width * 100,
                        'y': y_min / img_height * 100,
                        'width': (x_max - x_min) / img_width * 100,
                        'height': (y_max - y_min) / img_height * 100
                    },
                    'score': confidence
                })

                all_scores.append(confidence)

                avg_score = sum(all_scores) / max(len(all_scores), 1)

        return [{
            'result': results,
            'score': avg_score
        }]

        print(f'{image_url} -------- {image_path} =========== {img} ------------- {img_height, img_width} ----{index}')
        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        return []

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        task = self.download_labeled_tasks(4)
        print("-----------*******************")
        print(task)

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')



