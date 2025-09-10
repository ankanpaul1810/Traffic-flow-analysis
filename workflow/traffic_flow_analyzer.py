import cv2
import numpy as np
import pandas as pd
import torch
import yt_dlp
import os
import time
from datetime import datetime
import argparse
from collections import defaultdict
import json

from ultralytics import YOLO
from sort import Sort

class LaneDefiner:
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lanes = self._define_lanes()
    
    def _define_lanes(self):
        lane_width = self.frame_width // 3
        lanes = {}
        lanes[1] = {
            'points': np.array([
                [0, self.frame_height],
                [lane_width, self.frame_height],
                [lane_width * 0.8, self.frame_height * 0.3],
                [lane_width * 0.2, self.frame_height * 0.3]
            ], dtype=np.int32),
            'color': (0, 255, 0), 
            'name': 'Lane 1 (Left)'
        }
        
        lanes[2] = {
            'points': np.array([
                [lane_width, self.frame_height],
                [lane_width * 2, self.frame_height],
                [lane_width * 1.8, self.frame_height * 0.3],
                [lane_width * 1.2, self.frame_height * 0.3]
            ], dtype=np.int32),
            'color': (255, 0, 0), 
            'name': 'Lane 2 (Center)'
        }
        
        lanes[3] = {
            'points': np.array([
                [lane_width * 2, self.frame_height],
                [self.frame_width, self.frame_height],
                [self.frame_width * 0.9, self.frame_height * 0.3],
                [lane_width * 1.8, self.frame_height * 0.3]
            ], dtype=np.int32),
            'color': (0, 0, 255),
            'name': 'Lane 3 (Right)'
        }
        
        return lanes
    
    def get_lane_for_point(self, x, y):
        """Determine which lane a point belongs to"""
        point = (int(x), int(y))
        
        for lane_id, lane_data in self.lanes.items():
            if cv2.pointPolygonTest(lane_data['points'], point, False) >= 0:
                return lane_id
        
        return 0
    
    def draw_lanes(self, frame):
        overlay = frame.copy()
        
        for lane_id, lane_data in self.lanes.items():
            cv2.fillPoly(overlay, [lane_data['points']], lane_data['color'])
            cv2.polylines(frame, [lane_data['points']], True, lane_data['color'], 2)
        
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        return frame

class VehicleCounter:    
    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.create_output_directory()
        
        self.model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        self.vehicle_classes = [2, 3, 5, 7]
        
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.lane_definer = LaneDefiner(self.frame_width, self.frame_height)
        self.vehicle_data = []
        self.lane_counts = {1: 0, 2: 0, 3: 0}
        self.tracked_vehicles = {}
        self.frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.output_dir, 'analyzed_traffic.mp4')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                 (self.frame_width, self.frame_height))
    
    def create_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls in self.vehicle_classes and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections)
    
    def update_tracking(self, detections):
        if len(detections) > 0:
            tracked_objects = self.tracker.update(detections)
        else:
            tracked_objects = self.tracker.update(np.empty((0, 5)))
        
        return tracked_objects
    
    def process_frame(self, frame):
        self.frame_count += 1
        timestamp = self.frame_count / self.fps        
        detections = self.detect_vehicles(frame)
        tracked_objects = self.update_tracking(detections)
        frame = self.lane_definer.draw_lanes(frame)
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            lane_id = self.lane_definer.get_lane_for_point(center_x, center_y)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
 
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = {
                    'first_seen_frame': self.frame_count,
                    'lane': lane_id,
                    'counted': False
                }
                
                if lane_id > 0 and not self.tracked_vehicles[track_id]['counted']:
                    self.lane_counts[lane_id] += 1
                    self.tracked_vehicles[track_id]['counted'] = True
                    
                    self.vehicle_data.append({
                        'Vehicle_ID': track_id,
                        'Lane': lane_id,
                        'Frame': self.frame_count,
                        'Timestamp': round(timestamp, 2),
                        'Center_X': center_x,
                        'Center_Y': center_y
                    })
        
        self.draw_lane_counts(frame)
        
        return frame
    
    def draw_lane_counts(self, frame):
        y_offset = 50
        for lane_id in [1, 2, 3]:
            count = self.lane_counts[lane_id]
            lane_name = self.lane_definer.lanes[lane_id]['name']
            color = self.lane_definer.lanes[lane_id]['color']
            
            text = f"{lane_name}: {count}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2)
            y_offset += 40
        
        total_count = sum(self.lane_counts.values())
        cv2.putText(frame, f"Total: {total_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def save_results(self):
        df = pd.DataFrame(self.vehicle_data)
        csv_path = os.path.join(self.output_dir, 'vehicle_counts.csv')
        df.to_csv(csv_path, index=False)
        
        summary = {
            'total_vehicles': sum(self.lane_counts.values()),
            'lane_counts': self.lane_counts,
            'total_frames': self.frame_count,
            'processing_date': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return csv_path, summary_path
    
    def process_video(self):
        print("Starting traffic flow analysis...")
        print(f"Video dimensions: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps}")
        
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            
            self.out.write(processed_frame)
            
            if self.frame_count % 30 == 0: 
                elapsed = time.time() - start_time
                fps_current = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {self.frame_count}, FPS: {fps_current:.2f}, "
                      f"Total vehicles: {sum(self.lane_counts.values())}")
            
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        csv_path, summary_path = self.save_results()
        
        print("\n" + "="*50)
        print("TRAFFIC FLOW ANALYSIS COMPLETE")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")
        print(f"Average FPS: {self.frame_count / (time.time() - start_time):.2f}")
        print("\nVEHICLE COUNTS BY LANE:")
        for lane_id, count in self.lane_counts.items():
            lane_name = self.lane_definer.lanes[lane_id]['name']
            print(f"  {lane_name}: {count}")
        print(f"  Total vehicles: {sum(self.lane_counts.values())}")
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  Summary: {summary_path}")
        print(f"  Video: {os.path.join(self.output_dir, 'analyzed_traffic.mp4')}")

def download_youtube_video(url, output_path="input_video.mp4"):
    """Download video from YouTube"""
    print(f"Downloading video from: {url}")
    
    ydl_opts = {
        'format': 'best[height<=720]',  
        'outtmpl': output_path.replace('.mp4', '.%(ext)s'),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            base_name = output_path.replace('.mp4', '')
            for ext in ['.mp4', '.webm', '.mkv']:
                if os.path.exists(base_name + ext):
                    if ext != '.mp4':
                        os.rename(base_name + ext, output_path)
                    return output_path
            return output_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Traffic Flow Analysis System')
    parser.add_argument('--video', type=str, 
                       default='https://www.youtube.com/watch?v=MNn9qKG2UFI',
                       help='Video URL or local path')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.video.startswith('http'):
        video_path = download_youtube_video(args.video, "input_video.mp4")
        if video_path is None:
            print("Failed to download video. Exiting.")
            return
    else:
        video_path = args.video
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    counter = VehicleCounter(video_path, args.output)
    counter.process_video()

if __name__ == "__main__":
    main()