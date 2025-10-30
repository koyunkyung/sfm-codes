# Structure from Motion(SfM) Implementation

The objective of this code repository is to build an incremental Structure-from-Motion (SfM) system from scratch using Python and OpenCV. You will reconstruct 3D scenes from a video, compare your implementation with COLMAP, and visualize the camera trajectory and 3D points.

This repository includes:
- A custom-built incremental SfM pipeline
- A COLMAP-based baseline pipeline for comparison
- Tools for feature detection, matching, triangulation, and bundle adjustment
- 3D visualization of reconstructed point clouds and cameras
---

## Execution Flow

### Step 1: Frame Extraction
```bash
python data_loader.py
```
Extracts frames from a video (default: `data/video.mp4`)
Saves output frames to: `data/frames/`

### Step 2: SfM Pipeline
```bash
python my_sfm/main.py
```
This runs the incremental SfM pipeline with the following stages:
1. Feature Matching Verification (optional prompt)
2. Feature Extraction using SIFT
3. Graph Construction of image feature matches
4. Initial Pose Estimation via essential matrix and triangulation
5. Incremental Camera Registration using PnP
6. Triangulation of new points
7. Bundle Adjustment for refinement

**Outputs:**
```
results/my_sfm/
├── cameras.txt              # Camera intrinsics and extrinsics
├── images.txt               # Camera poses
├── points3D.txt             # 3D points with RGB
├── vis.txt                  # For 3D visualization
└── visualization.png        # Rendered point cloud
```

### Step 3: COLMAP Baseline Comparison
```bash
python colmap_baseline.py
```

**Outputs:**
```
results/colmap/
├── cameras.txt
├── images.txt
├── points3D.txt
├── vis.txt
└── visualization.png
```

### Visualization
Both pipelines generate a visualization.png file showing the reconstructed 3D point cloud.
- The custom pipeline uses Matplotlib 3D plots
- COLMAP results are converted to TXT and visualized similarly

---

## Project Structure
```
├── data_loader.py            # Frame extraction from video
├── my_sfm/
│   ├── main.py                   # Main entry for incremental SfM pipeline
│   ├── sfm_model.py              # Core SfM logic (triangulation, pose, BA)
│   └── feature.py                # SIFT-based feature extraction and matching
├── colmap_baseline.py        # COLMAP-based baseline pipeline
├── data/
│   ├── video.mp4             # Your input video (to be recorded)
│   └── frames/
├── results/
│   ├── my_sfm/               # Outputs from your SfM
│   └── colmap/               # Outputs from COLMAP

```
