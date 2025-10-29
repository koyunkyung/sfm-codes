import cv2
import numpy as np
from pathlib import Path

class VideoProcessor:
    
    def __init__(self, video_path, output_dir='data/frames', skip_frames=10):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.skip_frames = skip_frames
        self.frames = []
        self.frame_names = []
        
    def extract_frames(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        frame_count = 0
        saved_count = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # skip_frames마다 저장
            if frame_count % self.skip_frames == 0:
                frame_name = f"frame_{saved_count:04d}.jpg"
                frame_path = self.output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                
                self.frames.append(frame)
                self.frame_names.append(frame_name)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"✅ Extracted {saved_count} frames (skipped every {self.skip_frames})")
        return self.frames, self.frame_names
    
    def load_frames(self):
        frame_paths = sorted(self.output_dir.glob("*.jpg"))
        
        for path in frame_paths:
            frame = cv2.imread(str(path))
            self.frames.append(frame)
            self.frame_names.append(path.name)
            
        print(f"📂 Loaded {len(self.frames)} frames")
        return self.frames, self.frame_names
    
if __name__ == "__main__":
    import sys
    processor = VideoProcessor('data/video.mp4', skip_frames=10)
    frames, names = processor.extract_frames()