from ultralytics import YOLO
from celery import Celery
from datetime import datetime

# Create a Celery instance
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task(bind=True)
def trainyolov8(self ,custom_yaml_path):
    """Task to train YOLOv8"""
    try:
        # Optionally, update state before starting
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100})

        # Load a model
        model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # Load pretrained model

        # Get the current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create the message
        message = f"progress time is ({current_time})"

        # Save the message to a text file
        with open("filename.txt", "w") as f:
            f.write(current_time)

        # Train the model
        results = model.train(data=custom_yaml_path, epochs=2, imgsz=640, project="ml_backend_yolov8test")

        self.update_state(state='SUCCESS', meta={'current': 100, 'total': 100})
        return {'current': 100, 'total': 100, 'status': 'Task completed!'}
    except Exception as e:
        # Log any exceptions
        print(f"An error occurred: {str(e)}")
        self.update_state(state='FAILURE', meta={'error_message': str(e)})
        return {'current': 0, 'total': 100, 'status': 'Task failed!'}