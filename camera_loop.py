import cv2
import numpy as np
import onnxruntime as ort

# Define the standard 17 YOLOv8 keypoint connections (skeleton lines)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Facial keypoints (Nose, Eyes, Ears)
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left Arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                     # Right Arm (Shoulder -> Elbow -> Wrist)
    (5, 11), (6, 12), (11, 12),          # Torso (Shoulders to Hips)
    (11, 13), (13, 15),                  # Left Leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16)                   # Right Leg (Hip -> Knee -> Ankle)
]

def preprocess(frame, input_size=(640, 640)):
    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, orig_w, orig_h

def main():
    model_path = "yolov8n-pose.onnx"
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Full structural tracking activated...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        input_tensor, orig_w, orig_h = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        
        preds = np.squeeze(outputs[0]).T
        confidences = preds[:, 4]
        best_idx = np.argmax(confidences)
        
        if confidences[best_idx] > 0.3:
            best_pred = preds[best_idx]
            kpts = best_pred[5:] # 17 keypoints x 3 elements (x, y, conf) = 51 values
            
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0

            # Dictionary to temporarily store mapped (x, y) coordinates for drawing links
            coord_map = {}

            # 1. Map and draw all 17 individual joint points
            for i in range(17):
                idx = i * 3
                x = kpts[idx]
                y = kpts[idx + 1]
                conf = kpts[idx + 2]

                if conf > 0.3:
                    screen_x = int(x * scale_x)
                    screen_y = int(y * scale_y)
                    coord_map[i] = (screen_x, screen_y)
                    
                    # Draw a cyan circle on every visible joint node
                    cv2.circle(frame, (screen_x, screen_y), 5, (255, 255, 0), -1)

            # 2. Connect the dots by drawing the skeletal connection lines
            for start_joint, end_joint in SKELETON_CONNECTIONS:
                if start_joint in coord_map and end_joint in coord_map:
                    pt1 = coord_map[start_joint]
                    pt2 = coord_map[end_joint]
                    # Draw a glowing green bone link segment
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(frame, "NPU Cyber Canvas - Full Rig Active", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Cyber Canvas Tracking Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()