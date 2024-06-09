from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import os, sys, random, shutil
import label_studio_sdk
from label_studio_ml.utils import get_image_size, get_single_tag_keys, is_skipped
from label_studio.core.utils.io import  get_data_dir
from PIL import Image
from dotenv import load_dotenv
from train_yolo import trainyolov8
from rq import Queue, Worker, Connection
from redis import Redis



IMG_DATA = 'my_ml_backyolotest/data/images/'
LABEL_DATA = 'my_ml_backyolotest/data/labels/'

# Function to create directory if it does not exist
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

# Check and create directories
create_dir_if_not_exists(IMG_DATA)
create_dir_if_not_exists(LABEL_DATA)


load_dotenv()
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")


queue = Queue(connection=Redis())


if not LABEL_STUDIO_HOST or not LABEL_STUDIO_API_KEY:
    raise ValueError("LABEL_STUDIO_HOST and LABEL_STUDIO_API_KEY must be set in the .env file")


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def __init__(self, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        
        
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')

        self.hostname = LABEL_STUDIO_HOST
        self.access_token = LABEL_STUDIO_API_KEY
        
        self.image_dir = upload_dir

        
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
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")
    

    def reset_train_dir(self, dir_path):
        #remove cache file and reset train/val dir
        if os.path.isfile(os.path.join(dir_path,"train.cache")):
            os.remove(os.path.join(LABEL_DATA, "train.cache"))
            os.remove(os.path.join(LABEL_DATA, "val.cache"))

        for dir in os.listdir(dir_path):
            shutil.rmtree(os.path.join(dir_path, dir))
            os.makedirs(os.path.join(dir_path, dir))
    
    def _get_image_url(self,task):
        image_url = task['data'][self.value]
        return image_url


    def download_tasks(self, project):
        """
        Download all labeled tasks from project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/
        :param project: project ID
        :return:
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project)
        tasks = project.get_labeled_tasks()
        return tasks
    

    def _get_image_url(self,task):
        image_url = task['data'][self.value]
        return image_url

    def extract_data_from_tasks(self, tasks):
        img_labels = []
        for task in tasks:
                
            if is_skipped(task):
                continue
                        
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            image_name = image_path.split("\\")[-1]
            Image.open(image_path).save(IMG_DATA+image_name)

            img_labels.append(task['annotations'][0]['result'])

            for annotation in task['annotations']:
                for bbox in annotation['result']:
                    bb_width = (bbox['value']['width']) / 100
                    bb_height = (bbox['value']['height']) / 100
                    x = (bbox['value']['x'] / 100 ) + (bb_width/2)
                    y = (bbox['value']['y'] / 100 ) + (bb_height/2)
                    label = bbox['value']['rectanglelabels']

                    #you need to get the label idx later on 
                    label_idx = 0
                        
                    with open(LABEL_DATA+image_name[:-4]+'.txt', 'a') as f:
                        f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")    
        

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

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
        
        return ModelResponse(predictions=[])
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        print("--------------------------------------------------Start training--------------------------------------------------------")
        print("===============================================line 1 is perfect================================================================")
        print(f"=========================the current working dir is==================={os. getcwd()}===================================")

        # for dir_path in [IMG_DATA, LABEL_DATA]:
        #     self.reset_train_dir(dir_path)
        










        #===================================optimize the loop becuse whenever the update or submit is clicked the entire process is running which takes more time==================




        print(f"===========daata  ========================= {data}")
        if data:
            #data = kwargs['data']
            project = data['project']['id']
            print(f"=======================project is this {project}")
            tasks = self.download_tasks(project)   
            self.extract_data_from_tasks(tasks) 
        else:
            self.extract_data_from_tasks(tasks)

        # use cache to retrieve the data from the previous fit() runs
        print(f"========================print the data===================================================================") 





       #======================================implement the train in the rq worker which shoul run in the background==========================================================================
    
        customyamlpath = "ml_backend_yolov8test\custum.yaml"
        #trainyolov8(custom_yaml_path= customyamlpath)
        job = queue.enqueue(trainyolov8, customyamlpath)
        print(f"Training job enqueued with ID {job.id}")


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

