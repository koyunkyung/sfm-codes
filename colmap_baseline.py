import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


class COLMAPBaseline:
    
    def __init__(self, image_dir='data/frames', output_dir='results/colmap'):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.database_path = self.output_dir / 'database.db'
        self.model_dir = self.output_dir / 'sparse' / '0'
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):

        import os
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        
        steps = [
            ('Feature Extraction', [
                'colmap', 'feature_extractor',
                '--database_path', str(self.database_path),
                '--image_path', str(self.image_dir),
                '--ImageReader.single_camera', '1',
                '--ImageReader.camera_model', 'OPENCV',
                '--SiftExtraction.use_gpu', '0'
            ]),
            ('Feature Matching', [
                'colmap', 'exhaustive_matcher',
                '--database_path', str(self.database_path),
                '--SiftMatching.use_gpu', '0'
            ]),
            ('Incremental Mapping', [
                'colmap', 'mapper',
                '--database_path', str(self.database_path),
                '--image_path', str(self.image_dir),
                '--output_path', str(self.model_dir.parent)
            ]),
            ('Model Conversion', [
                'colmap', 'model_converter',
                '--input_path', str(self.model_dir),
                '--output_path', str(self.model_dir),
                '--output_type', 'TXT'
            ])
        ]
        
        for i, (name, cmd) in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {name}...")
            subprocess.run(cmd, check=True, env=env, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("\n✅ COLMAP Pipeline Completed\n")
    
    def load_results(self):
        cameras = self._load_cameras()
        images = self._load_images()
        points3D = self._load_points3D()
        
        print(f"Reconstruction Summary:")
        print(f"   Cameras: {len(cameras)}")
        print(f"   Images:  {len(images)}")
        print(f"   Points:  {len(points3D)}\n")
        
        return cameras, images, points3D
    
    def _load_cameras(self):
        cameras = {}
        with open(self.model_dir / 'cameras.txt', 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                cameras[int(parts[0])] = {
                    'model': parts[1],
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'params': np.array([float(x) for x in parts[4:]])
                }
        return cameras
    
    def _load_images(self):
        """Load camera poses"""
        images = {}
        with open(self.model_dir / 'images.txt', 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
        
        for i in range(0, len(lines), 2):
            parts = lines[i].strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            t = np.array([float(x) for x in parts[5:8]])
            
            images[image_id] = {
                'name': parts[9],
                'R': self._quat2rot(qw, qx, qy, qz),
                't': t,
                'camera_id': int(parts[8])
            }
        
        return images
    
    def _load_points3D(self):
        points3D = {}
        with open(self.model_dir / 'points3D.txt', 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                points3D[int(parts[0])] = {
                    'xyz': np.array([float(x) for x in parts[1:4]]),
                    'rgb': np.array([int(x) for x in parts[4:7]]),
                    'error': float(parts[7])
                }
        return points3D
    
    def _quat2rot(self, w, x, y, z):
        """Convert quaternion to rotation matrix"""
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
    def generate_vis_file(self, points3D):
        """Generate vis.txt in example format: x y z R G B"""
        vis_path = self.model_dir / 'vis.txt'
        
        with open(vis_path, 'w') as f:
            for pt in points3D.values():
                xyz, rgb = pt['xyz'], pt['rgb']
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
        
        print(f"Generated vis.txt ({len(points3D)} points)")
        return vis_path
    
    def visualize(self, vis_path):
        data = np.loadtxt(vis_path)
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0
        
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor('#2d3561')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2d3561')
        
        # Plot point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=1, alpha=0.6)
        
        # Add reference circles
        center = np.mean(points, axis=0)
        theta = np.linspace(0, 2*np.pi, 100)
        
        for plane_idx, (i, j, k) in enumerate([(0,1,2), (0,2,1), (1,2,0)]):
            radius = np.std(points[:, [i,j]]) * 2
            circle = np.zeros((100, 3))
            circle[:, i] = center[i] + radius * np.cos(theta)
            circle[:, j] = center[j] + radius * np.sin(theta)
            circle[:, k] = center[k]
            ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], 
                   'w-', alpha=0.3, linewidth=1)
        
        # Styling
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('white')
        
        ax.grid(True, alpha=0.2, color='white')
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        output_path = self.model_dir / 'visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='#2d3561')
        plt.close()
        return output_path


def main():
    baseline = COLMAPBaseline(
        image_dir='data/frames',
        output_dir='results/colmap'
    )
    # Execute pipeline
    baseline.run()
    # Load results
    cameras, images, points3D = baseline.load_results()
    # Generate outputs
    vis_path = baseline.generate_vis_file(points3D)
    baseline.visualize(vis_path)
    


if __name__ == "__main__":
    main()