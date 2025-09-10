"""Demo Script"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

def create_test_video(output_path="test_traffic.mp4", duration=10, fps=30):    
    print("Creating test video...")
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_frames = duration * fps
    
    vehicles = [
        {'x': 50, 'y': 400, 'w': 60, 'h': 40, 'dx': 3, 'dy': 0, 'color': (0, 255, 0)},
        {'x': 200, 'y': 350, 'w': 80, 'h': 50, 'dx': 4, 'dy': 0, 'color': (255, 0, 0)},
        {'x': 400, 'y': 300, 'w': 70, 'h': 45, 'dx': 2, 'dy': 0, 'color': (0, 0, 255)},
        {'x': 600, 'y': 250, 'w': 90, 'h': 55, 'dx': 5, 'dy': 0, 'color': (255, 255, 0)},
    ]
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, height//3), (width, height), (50, 50, 50), -1)

        for i in range(1, 3):
            x = width // 3 * i
            cv2.line(frame, (x, height//3), (x, height), (255, 255, 255), 2)
        
        for vehicle in vehicles:
            vehicle['x'] += vehicle['dx']
            
            if vehicle['x'] > width:
                vehicle['x'] = -vehicle['w']
            
            x1, y1 = int(vehicle['x']), int(vehicle['y'])
            x2, y2 = x1 + vehicle['w'], y1 + vehicle['h']
            
            if x1 < width and x2 > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), vehicle['color'], -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_num}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")
    return output_path

def run_demo():
    """Run a complete demo of the traffic flow analysis system"""
    
    print("="*60)
    print("TRAFFIC FLOW ANALYSIS SYSTEM - DEMO")
    print("="*60)
    print(f"Demo started at: {datetime.now()}")
    print()
    
    # Check if main script exists
    main_script = "traffic_flow_analyzer.py"
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found!")
        print("Please ensure you're in the correct directory.")
        return False
    
    print("Main script found")    
    print("\n Checking required packages...")
    
    required_packages = [
        'cv2', 'numpy', 'pandas', 'torch'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f" Yes {package}")
        except ImportError:
            print(f" No {package}")
            missing_packages.append(package)
    
    try:
        from ultralytics import YOLO
        print(f"  yes ultralytics")
    except ImportError:
        print(f"  no ultralytics")
        missing_packages.append('ultralytics')
    
    if missing_packages:
        print(f"\n Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n All packages available")
    
    print("\n Creating test video...")
    test_video = create_test_video("demo_traffic.mp4", duration=5, fps=15)
    
    print("\n Testing video processing")
    cap = cv2.VideoCapture(test_video)
    
    if not cap.isOpened():
        print("Could not open test video")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f" Video specs: {width}x{height}, {frame_count} frames, {fps} FPS")
    
    frames_read = 0
    while frames_read < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
    
    cap.release()
    
    if frames_read > 0:
        print(f" Successfully read {frames_read} frames")
    else:
        print(" Error: Could not read video frames")
        return False
    
    print("\n Testing YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  
        print(" YOLO model loaded successfully")
        
        cap = cv2.VideoCapture(test_video)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            results = model(frame, verbose=False)
            print(f" Detection test completed")
        
    except Exception as e:
        print(f" YOLO model error: {str(e)}")
        return False
    
    if os.path.exists(test_video):
        os.remove(test_video)
        print(f"\n🧹 Cleaned up test video")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n Next Steps:")
    print("1. Run the full analysis:")
    print("   python traffic_flow_analyzer.py")
    print("\n2. Use custom video:")
    print("   python traffic_flow_analyzer.py --video 'your_video.mp4'")
    print("\n3. Check output folder for results:")
    print("   - vehicle_counts.csv")
    print("   - analyzed_traffic.mp4")
    print("   - summary.json")
    
    return True

def check_system_info():
    """Display system information for troubleshooting"""
    print("\n System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("  PyTorch: Not installed")
    
    try:
        import cv2
        print(f"  OpenCV version: {cv2.__version__}")
    except ImportError:
        print("  OpenCV: Not installed")

if __name__ == "__main__":
    print("Starting Traffic Flow Analysis Demo...")
    
    check_system_info()
    success = run_demo()
    
    if not success:
        print("\n Demo failed. Please check the error messages above.")
        print("For help, refer to the README.md file or create an issue on GitHub.")
        sys.exit(1)
    else:
        print("\n System is ready for traffic flow analysis!")
        sys.exit(0)