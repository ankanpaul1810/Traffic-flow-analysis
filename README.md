# Traffic Flow Analysis System

A comprehensive computer vision solution for analyzing traffic flow by counting vehicles in three distinct lanes using YOLO object detection and SORT tracking algorithms.

## 🚀 Features

- **Real-time Vehicle Detection**: Uses YOLOv8 pre-trained COCO model for accurate vehicle detection
- **Multi-lane Tracking**: Automatically defines and tracks vehicles across three distinct lanes
- **Vehicle Counting**: Prevents duplicate counting using SORT (Simple Online and Realtime Tracking)
- **YouTube Integration**: Downloads and processes videos directly from YouTube URLs
- **Comprehensive Output**: Generates CSV files, summary statistics, and annotated video output
- **Performance Optimized**: Designed for real-time or near real-time processing on standard hardware

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 8GB RAM (16GB recommended)
- GPU support recommended (CUDA-compatible) for better performance

### Python Dependencies
All dependencies are listed in `requirements.txt`. Key packages include:
- OpenCV for video processing
- Ultralytics YOLOv8 for object detection
- PyTorch for deep learning
- FilterPy for Kalman filtering
- yt-dlp for YouTube video downloading
- Pandas for data processing

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd traffic-flow-analysis
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python traffic_flow_analyzer.py --help
```

## 🚦 Usage

### Basic Usage
Process the default YouTube traffic video:
```bash
python traffic_flow_analyzer.py
```

### Custom Video URL
```bash
python traffic_flow_analyzer.py --video "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Local Video File
```bash
python traffic_flow_analyzer.py --video "path/to/your/video.mp4"
```

### Custom Output Directory
```bash
python traffic_flow_analyzer.py --output "custom_output_folder"
```

### Complete Command with All Options
```bash
python traffic_flow_analyzer.py \
    --video "https://www.youtube.com/watch?v=MNn9qKG2UFI" \
    --output "results"
```

## 📊 Output Files

The system generates several output files in the specified output directory:

### 1. `vehicle_counts.csv`
Contains detailed tracking data for each vehicle:
```csv
Vehicle_ID,Lane,Frame,Timestamp,Center_X,Center_Y
1,2,45,1.5,320,240
2,1,67,2.23,150,220
...
```

### 2. `analyzed_traffic.mp4`
Annotated video showing:
- Lane boundaries (colored polygons)
- Vehicle bounding boxes
- Vehicle tracking IDs
- Real-time lane counts
- Total vehicle count

### 3. `summary.json`
Summary statistics including:
```json
{
  "total_vehicles": 127,
  "lane_counts": {
    "1": 42,
    "2": 51,
    "3": 34
  },
  "total_frames": 1800,
  "processing_date": "2025-09-10T14:30:00"
}
```

## 🏗️ System Architecture

### Core Components

1. **LaneDefiner Class**
   - Defines three lane boundaries as polygons
   - Determines vehicle lane assignment
   - Renders lane visualizations

2. **VehicleCounter Class**
   - Main processing engine
   - Coordinates detection, tracking, and counting
   - Manages output generation

3. **YOLO Detection**
   - Uses YOLOv8n (nano) for speed/accuracy balance
   - Filters for vehicle classes (car, motorcycle, bus, truck)
   - Confidence threshold filtering

4. **SORT Tracking**
   - Kalman filter-based tracking
   - Handles object association across frames
   - Prevents duplicate counting

### Lane Definition Logic

The system automatically defines three lanes based on video dimensions:
- **Lane 1 (Left)**: Green boundary
- **Lane 2 (Center)**: Blue boundary  
- **Lane 3 (Right)**: Red boundary

Lanes are defined as trapezoidal polygons to match typical road perspective.

### Vehicle Counting Algorithm

1. **Detection**: YOLO detects vehicles in each frame
2. **Tracking**: SORT associates detections across frames
3. **Lane Assignment**: Vehicle center point determines lane membership
4. **Counting**: First appearance in valid lane triggers count increment
5. **Duplicate Prevention**: Each track ID counted only once per lane

## ⚙️ Configuration

### Adjustable Parameters

You can modify these parameters in the source code:

```python
# Vehicle detection confidence threshold
conf_threshold = 0.5

# SORT tracker parameters
max_age = 30        # Max frames to keep tracker without detection
min_hits = 3        # Min detections before confirming track
iou_threshold = 0.3 # IoU threshold for association

# Vehicle classes (COCO dataset)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
```

### Lane Customization

To adjust lane boundaries, modify the `_define_lanes()` method in the `LaneDefiner` class:

```python
def _define_lanes(self):
    # Customize lane polygon points here
    lanes[1]['points'] = np.array([...])  # Your custom points
```

## 🎯 Performance Optimization

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support for faster inference
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB or more for large videos
- **Storage**: SSD for faster I/O operations

### Software Optimizations
- Use YOLOv8n (nano) for speed vs YOLOv8s/m/l for accuracy
- Adjust video resolution for processing speed
- Enable GPU acceleration in PyTorch
- Use appropriate batch sizes for your hardware

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip uninstall opencv-python
   pip install opencv-python
   ```

2. **CUDA/GPU Issues**
   ```python
   # Force CPU usage
   import torch
   torch.cuda.is_available = lambda: False
   ```

3. **Video Download Failures**
   - Check internet connection
   - Verify YouTube URL accessibility
   - Try alternative video URLs

4. **Memory Issues**
   - Reduce video resolution
   - Process shorter video segments
   - Increase system RAM

### Performance Issues
- **Low FPS**: Reduce video resolution, use YOLOv8n, enable GPU
- **Inaccurate Counts**: Adjust confidence threshold, modify lane boundaries
- **Missing Detections**: Lower confidence threshold, check lighting conditions

## 📈 Evaluation Metrics

The system tracks several performance metrics:

- **Processing FPS**: Frames processed per second
- **Detection Accuracy**: Percentage of correctly identified vehicles
- **Tracking Consistency**: Maintenance of vehicle IDs across frames
- **Lane Assignment Accuracy**: Correct lane classification percentage

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- **YOLO**: Ultralytics for YOLOv8 implementation
- **SORT**: Alex Bewley for SORT tracking algorithm
- **OpenCV**: Open Source Computer Vision Library
- **PyTorch**: Deep learning framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed description
4. Include system information and error logs

**Note**: This system is designed for traffic analysis and research purposes. Ensure compliance with privacy and data protection regulations when processing real traffic footage.
