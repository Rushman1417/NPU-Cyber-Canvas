import cv2
import numpy as np
import onnxruntime as ort
import time
from camera_loop import WebcamVideoStream, preprocess # Reuse your threaded webcam class

def apply_distortion(frame, pt1, pt2):
    h, w = frame.shape[:2]
    x_min = max(0, min(pt1[0], pt2[0]) - 50)
    y_min = max(0, min(pt1[1], pt2[1]) - 50)
    x_max = min(w, max(pt1[0], pt2[0]) + 50)
    y_max = min(h, max(pt1[1], pt2[1]) + 50)
    
    if x_max - x_min > 10 and y_max - y_min > 10:
        roi = frame[y_min:y_max, x_min:x_max]
        distorted_roi = cv2.bitwise_not(roi)
        distorted_roi = cv2.cvtColor(distorted_roi, cv2.COLOR_BGR2XYZ)
        distorted_roi = cv2.cvtColor(distorted_roi, cv2.COLOR_XYZ2BGR)
        frame[y_min:y_max, x_min:x_max] = cv2.addWeighted(roi, 0.4, distorted_roi, 0.6, 0)
        cv2.line(frame, pt1, pt2, (255, 0, 255), 3, cv2.LINE_AA)
    return frame

def run_distortion(session, input_name):
    vvs = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    print("Spatial Distortion Engine running... Press 'q' to go back.")

    while True:
        frame = vvs.read()
        if frame is None: continue
        display_frame = frame.copy()
        input_tensor, orig_w, orig_h = preprocess(frame)

        outputs = session.run(None, {input_name: input_tensor})
        preds = np.squeeze(outputs[0]).T
        best_idx = np.argmax(preds[:, 4])
        
        if preds[best_idx, 4] > 0.3:
            kpts = preds[best_idx, 5:]
            scale_x, scale_y = orig_w / 640.0, orig_h / 640.0
            
            lw_x, lw_y, lw_conf = kpts[9*3], kpts[9*3+1], kpts[9*3+2]
            rw_x, rw_y, rw_conf = kpts[10*3], kpts[10*3+1], kpts[10*3+2]

            if lw_conf > 0.4 and rw_conf > 0.4:
                pt_left = (int(lw_x * scale_x), int(lw_y * scale_y))
                pt_right = (int(rw_x * scale_x), int(rw_y * scale_y))
                display_frame = apply_distortion(display_frame, pt_left, pt_right)
                cv2.circle(display_frame, pt_left, 10, (255, 0, 255), -1)
                cv2.circle(display_frame, pt_right, 10, (255, 0, 255), -1)

        cv2.putText(display_frame, "NPU Spatial Distortion Mode", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Cyber Canvas Workspace', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vvs.stop()
    cv2.destroyAllWindows()