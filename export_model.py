from ultralytics import YOLO

def main():
    # 1. Pull down the ultra-lightweight pose estimation model
    print("Loading yolov8n-pose.pt model...")
    model = YOLO("yolov8n-pose.pt")

    # 2. Export it directly to quantized ONNX format for embedded/ARM devices
    print("Exporting model to ONNX format...")
    path = model.export(format="onnx", imgsz=640)
    print(f"Model successfully exported to: {path}")

if __name__ == "__main__":
    main()
