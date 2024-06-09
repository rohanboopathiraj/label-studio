from ultralytics import YOLO

def trainyolov8(custom_yaml_path):
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=custom_yaml_path, epochs=2, imgsz=640, project = "ml_backend_yolov8test")
