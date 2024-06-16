from ultralytics import YOLO


def traintheyolov8(custom_yaml_path):
    # Load a model
    print("loading the model")
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    print("Model loading finished")
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights
    print("Model started finetuning")
    # Train the model
    results = model.train(data=custom_yaml_path, epochs=2, imgsz=640, project = "my_ml_backyolotest/models")

    