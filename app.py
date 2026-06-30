import os
import onnxruntime as ort
import sys

# Import our modular workflow modes
from distortion_mode import run_distortion
from paint_mode import run_paint

def main():
    print("==============================================")
    print("      INITIALIZING NPU CYBER CANVAS CORE      ")
    print("==============================================")
    
    # Initialize the common model infrastructure once at startup
    model_path = "yolov8n-pose.onnx"
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: {model_path} missing from project root folder.")
        return

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(model_path, opts)
    input_name = session.get_inputs()[0].name
    
    while True:
        print("\n[CHOOSE WORKSPACE ENVIRONMENT]")
        print("1 -> Launch Spatial Distortion Engine")
        print("2 -> Launch Interactive Paint Canvas")
        print("Q -> Shutdown System")
        
        choice = input("Select Environment Layer (1, 2, or Q): ").strip().upper()
        
        if choice == '1':
            run_distortion(session, input_name)
        elif choice == '2':
            run_paint(session, input_name)
        elif choice == 'Q':
            print("Shutting down Cyber Canvas engine modules gracefully.")
            break
        else:
            print("Invalid matrix input selection. Try again.")

if __name__ == "__main__":
    main()