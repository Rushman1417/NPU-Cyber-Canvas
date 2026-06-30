import cv2
import numpy as np
import time
from camera_loop import WebcamVideoStream, preprocess

def apply_rectangular_distortion(frame, pt1, pt2):
    """Creates a rectangular horizontal stretching distortion zone between hands."""
    h, w = frame.shape[:2]
    
    # Identify the exact horizontal bounds (left and right columns)
    x_min = max(0, min(pt1[0], pt2[0]))
    x_max = min(w, max(pt1[0], pt2[0]))
    
    # We want the effect to span a distinct vertical zone around your hands (e.g., center 70% of screen)
    y_min = max(0, min(pt1[1], pt2[1]) - 100)
    y_max = min(h, max(pt1[1], pt2[1]) + 100)
    
    # Ensure the box is wide enough to warp
    zone_width = x_max - x_min
    zone_height = y_max - y_min
    if zone_width < 20 or zone_height < 20: 
        return frame

    # 1. Extract the clean slice from the frame
    roi = frame[y_min:y_max, x_min:x_max]
    
    # 2. Generate an intentional horizontal sine-wave stretching map
    # This stretches the pixels horizontally inside the rectangular boundary columns
    map_x, map_y = np.meshgrid(np.arange(zone_width), np.arange(zone_height))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Calculate horizontal displacement relative to the width of the box
    # A sine wave across the X-axis stretches the center of the box while keeping edges locked
    displacement = 25.0 * np.sin((map_x / zone_width) * np.pi)
    map_x += displacement

    # 3. Remap the pixels smoothly within the rectangle box bounds
    distorted_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Overwrite the area inside the frame
    frame[y_min:y_max, x_min:x_max] = distorted_roi
    
    # Optional: Draw a subtle, thin neon border on the left and right edges ONLY (no solid connecting lines)
    cv2.line(frame, (x_min, y_min), (x_min, y_max), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_min), (x_max, y_max), (0, 255, 255), 1, cv2.LINE_AA)

    return frame

def run_distortion(session, input_name):
    vvs = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    print("Rectangular Distortion Layer Running...")

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
            
            # Left Wrist = index 9, Right Wrist = index 10
            lw_x, lw_y, lw_conf = kpts[9*3], kpts[9*3+1], kpts[9*3+2]
            rw_x, rw_y, rw_conf = kpts[10*3], kpts[10*3+1], kpts[10*3+2]

            if lw_conf > 0.4 and rw_conf > 0.4:
                pt_left = (int(lw_x * scale_x), int(lw_y * scale_y))
                pt_right = (int(rw_x * scale_x), int(rw_y * scale_y))
                display_frame = apply_rectangular_distortion(display_frame, pt_left, pt_right)

        cv2.putText(display_frame, "NPU Rectangular Warp Mode", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Cyber Canvas Workspace', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vvs.stop()
    cv2.destroyAllWindows()