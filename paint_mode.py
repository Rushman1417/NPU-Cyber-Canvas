import cv2
import numpy as np
import onnxruntime as ort
import time
from camera_loop import WebcamVideoStream, preprocess

def run_paint(session, input_name):
    vvs = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    
    # Create a blank black canvas layout to hold our continuous drawings
    paint_canvas = None
    last_brush_pt = None
    print("Interactive Canvas Paint running... Press 'q' to go back.")

    while True:
        frame = vvs.read()
        if frame is None: continue
        
        if paint_canvas is None:
            paint_canvas = np.zeros_like(frame)

        # Slowly fade old drawings out slightly over time for a glowing trail effect
        paint_canvas = cv2.addWeighted(paint_canvas, 0.98, paint_canvas, 0, 0)

        input_tensor, orig_w, orig_h = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        preds = np.squeeze(outputs[0]).T
        best_idx = np.argmax(preds[:, 4])
        
        if preds[best_idx, 4] > 0.3:
            kpts = preds[best_idx, 5:]
            scale_x, scale_y = orig_w / 640.0, orig_h / 640.0
            
            # Right Wrist = index 10
            rw_x, rw_y, rw_conf = kpts[10*3], kpts[10*3+1], kpts[10*3+2]

            if rw_conf > 0.4:
                brush_pt = (int(rw_x * scale_x), int(rw_y * scale_y))
                if last_brush_pt is not None:
                    # Draw a solid neon cyan paint stroke on the canvas
                    cv2.line(paint_canvas, last_brush_pt, brush_pt, (255, 255, 0), 6, cv2.LINE_AA)
                last_brush_pt = brush_pt
            else:
                last_brush_pt = None
        else:
            last_brush_pt = None

        # Blend your live camera feed together with the persistent digital paint strokes
        display_frame = cv2.addWeighted(frame, 0.7, paint_canvas, 0.9, 0)

        cv2.putText(display_frame, "NPU Digital Paint Mode (Right Hand)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Cyber Canvas Workspace', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vvs.stop()
    cv2.destroyAllWindows()