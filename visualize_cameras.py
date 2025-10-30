"""
Camera Pose and Trajectory Visualization
Reads COLMAP-format results and creates accurate visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import struct


def read_cameras_txt(filepath):
    """Parse COLMAP cameras.txt format"""
    cameras = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_txt(filepath):
    """Parse COLMAP images.txt format"""
    images = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else f"image_{img_id}"
        
        images[img_id] = {
            'id': img_id,
            'qvec': np.array([qw, qx, qy, qz]),
            'tvec': np.array([tx, ty, tz]),
            'camera_id': cam_id,
            'name': name
        }
        
        i += 2  # Skip the keypoint line
    
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def get_camera_center(R, t):
    """
    Compute camera center in world coordinates
    Camera center C = -R^T @ t
    """
    return -R.T @ t


def read_points3D_txt(filepath):
    """Parse COLMAP points3D.txt format"""
    points = []
    colors = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            
            points.append([x, y, z])
            colors.append([r/255.0, g/255.0, b/255.0])
    
    return np.array(points), np.array(colors)


def visualize_cameras_and_trajectory(images_data, points3d, colors, title, ax, 
                                     camera_color='red', traj_color='red'):
    """
    Visualize camera poses and trajectory
    
    Args:
        images_data: dict of image data with qvec and tvec
        points3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        title: plot title
        ax: matplotlib 3D axis
        camera_color: color for cameras
        traj_color: color for trajectory
    """
    
    # Extract camera positions
    camera_centers = []
    camera_orientations = []
    
    for img_id in sorted(images_data.keys()):
        img = images_data[img_id]
        
        # Convert quaternion to rotation matrix
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']
        
        # Get camera center
        C = get_camera_center(R, t)
        camera_centers.append(C)
        camera_orientations.append(R)
    
    camera_centers = np.array(camera_centers)
    
    # Plot 3D points (sparse for clarity)
    if len(points3d) > 0:
        # Subsample for better visualization
        n_points = min(len(points3d), 5000)
        indices = np.random.choice(len(points3d), n_points, replace=False)
        ax.scatter(points3d[indices, 0], 
                  points3d[indices, 1], 
                  points3d[indices, 2],
                  c=colors[indices], 
                  s=0.5, 
                  alpha=0.3,
                  label='3D Points')
    
    # Plot camera trajectory
    if len(camera_centers) > 1:
        ax.plot(camera_centers[:, 0], 
               camera_centers[:, 1], 
               camera_centers[:, 2],
               color=traj_color, 
               linewidth=2.5, 
               alpha=0.8,
               linestyle='--',
               label='Trajectory')
    
    # Plot camera poses
    for i, (C, R) in enumerate(zip(camera_centers, camera_orientations)):
        # Camera center
        ax.scatter([C[0]], [C[1]], [C[2]], 
                  color=camera_color, 
                  s=150, 
                  marker='o',
                  edgecolors='white',
                  linewidths=2,
                  zorder=10)
        
        # Camera orientation (viewing direction = -Z axis in camera frame)
        cam_scale = 1.5
        
        # Camera coordinate axes in world frame
        x_axis = R.T[:, 0] * cam_scale  # Red - X
        y_axis = R.T[:, 1] * cam_scale  # Green - Y
        z_axis = R.T[:, 2] * cam_scale  # Blue - Z (viewing direction)
        
        # Draw camera axes
        ax.plot([C[0], C[0] + z_axis[0]], 
               [C[1], C[1] + z_axis[1]], 
               [C[2], C[2] + z_axis[2]], 
               color='blue', linewidth=2.5, alpha=0.8)
        
        # Label cameras
        if i % max(1, len(camera_centers) // 15) == 0:  # Label every few cameras
            ax.text(C[0], C[1], C[2], 
                   f'{i}', 
                   fontsize=9, 
                   color='white',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor=camera_color, 
                            edgecolor='white',
                            linewidth=1.5,
                            alpha=0.9))
    
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Set equal aspect ratio
    if len(points3d) > 0 and len(camera_centers) > 0:
        all_points = np.vstack([points3d, camera_centers])
    elif len(camera_centers) > 0:
        all_points = camera_centers
    else:
        return
    
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Style
    ax.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)


def visualize_trajectory_2d(images_data, title, ax, color='red'):
    """
    2D trajectory visualization (top and side views)
    
    Args:
        images_data: dict of image data
        title: subplot title
        ax: matplotlib axis (2D)
        color: trajectory color
    """
    
    camera_centers = []
    
    for img_id in sorted(images_data.keys()):
        img = images_data[img_id]
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']
        C = get_camera_center(R, t)
        camera_centers.append(C)
    
    camera_centers = np.array(camera_centers)
    
    # Plot trajectory with arrows
    ax.plot(camera_centers[:, 0], camera_centers[:, 1], 
           'o-', color=color, linewidth=2.5, markersize=8,
           markerfacecolor=color, markeredgecolor='white', 
           markeredgewidth=2, alpha=0.8, label='Camera Path')
    
    # Add direction arrows
    for i in range(len(camera_centers) - 1):
        if i % max(1, len(camera_centers) // 8) == 0:  # Show some arrows
            dx = camera_centers[i+1, 0] - camera_centers[i, 0]
            dy = camera_centers[i+1, 1] - camera_centers[i, 1]
            ax.arrow(camera_centers[i, 0], camera_centers[i, 1],
                    dx * 0.5, dy * 0.5,
                    head_width=0.3, head_length=0.3,
                    fc='yellow', ec='orange', linewidth=2, alpha=0.8)
    
    # Label key cameras
    for i in [0, len(camera_centers)//2, -1]:
        if i < len(camera_centers):
            ax.scatter(camera_centers[i, 0], camera_centers[i, 1],
                      s=150, c=color, marker='o', 
                      edgecolors='white', linewidths=2.5, zorder=10)
            ax.annotate(f'{i}', 
                       (camera_centers[i, 0], camera_centers[i, 1]),
                       fontsize=11, fontweight='bold', color='white',
                       bbox=dict(boxstyle='circle,pad=0.3', 
                                facecolor='black', 
                                edgecolor='white', 
                                linewidth=1.5),
                       ha='center', va='center')
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_facecolor('#f8f9fa')
    
    # Add statistics
    distances = np.linalg.norm(np.diff(camera_centers[:, :2], axis=0), axis=1)
    ax.text(0.02, 0.98, 
           f'Cameras: {len(camera_centers)}\n'
           f'Total dist: {distances.sum():.2f}m\n'
           f'Avg step: {distances.mean():.2f}m',
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def compare_reconstructions(colmap_dir, mysfm_dir, output_dir='comparison_results'):
    """
    Compare COLMAP and your SfM results with side-by-side visualization
    
    Args:
        colmap_dir: path to COLMAP sparse/0 directory
        mysfm_dir: path to your SfM results directory
        output_dir: where to save visualizations
    """
    
    print("\n" + "="*70)
    print("CAMERA POSE & TRAJECTORY VISUALIZATION")
    print("="*70 + "\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Load COLMAP results
    # ========================================================================
    print("Loading COLMAP results...")
    colmap_dir = Path(colmap_dir)
    
    colmap_images = read_images_txt(colmap_dir / 'images.txt')
    colmap_points, colmap_colors = read_points3D_txt(colmap_dir / 'points3D.txt')
    
    print(f"  COLMAP: {len(colmap_images)} cameras, {len(colmap_points)} points")
    
    # ========================================================================
    # Load Your SfM results
    # ========================================================================
    print("Loading Your SfM results...")
    mysfm_dir = Path(mysfm_dir)
    
    mysfm_images = read_images_txt(mysfm_dir / 'images.txt')
    mysfm_points, mysfm_colors = read_points3D_txt(mysfm_dir / 'points3D.txt')
    
    print(f"  Your SfM: {len(mysfm_images)} cameras, {len(mysfm_points)} points")
    
    # ========================================================================
    # 1. 3D Scene Comparison (Side by Side)
    # ========================================================================
    print("\nCreating 3D scene comparison...")
    
    fig = plt.figure(figsize=(20, 9))
    
    # COLMAP
    ax1 = fig.add_subplot(121, projection='3d')
    visualize_cameras_and_trajectory(
        colmap_images, colmap_points, colmap_colors,
        f'COLMAP Reconstruction\n({len(colmap_images)} cameras, {len(colmap_points)} points)',
        ax1,
        camera_color='#e74c3c',
        traj_color='#e74c3c'
    )
    
    # Your SfM
    ax2 = fig.add_subplot(122, projection='3d')
    visualize_cameras_and_trajectory(
        mysfm_images, mysfm_points, mysfm_colors,
        f'Your SfM Reconstruction\n({len(mysfm_images)} cameras, {len(mysfm_points)} points)',
        ax2,
        camera_color='#3498db',
        traj_color='#3498db'
    )
    
    fig.patch.set_facecolor('#16213e')
    plt.tight_layout()
    
    save_path = output_dir / '3d_comparison.png'
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    # ========================================================================
    # 2. Camera Trajectory Comparison (Top View)
    # ========================================================================
    print("Creating trajectory comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # COLMAP trajectory
    visualize_trajectory_2d(
        colmap_images,
        'COLMAP - Camera Trajectory (Top View)',
        axes[0],
        color='#e74c3c'
    )
    
    # Your SfM trajectory
    visualize_trajectory_2d(
        mysfm_images,
        'Your SfM - Camera Trajectory (Top View)',
        axes[1],
        color='#3498db'
    )
    
    plt.tight_layout()
    
    save_path = output_dir / 'trajectory_comparison.png'
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    # ========================================================================
    # 3. Overlay Trajectory Comparison
    # ========================================================================
    print("Creating overlay trajectory comparison...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Extract camera centers
    colmap_centers = []
    for img_id in sorted(colmap_images.keys()):
        img = colmap_images[img_id]
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']
        C = get_camera_center(R, t)
        colmap_centers.append(C)
    colmap_centers = np.array(colmap_centers)
    
    mysfm_centers = []
    for img_id in sorted(mysfm_images.keys()):
        img = mysfm_images[img_id]
        R = qvec2rotmat(img['qvec'])
        t = img['tvec']
        C = get_camera_center(R, t)
        mysfm_centers.append(C)
    mysfm_centers = np.array(mysfm_centers)
    
    # Plot both trajectories
    ax.plot(colmap_centers[:, 0], colmap_centers[:, 1],
           'o-', color='#e74c3c', linewidth=3, markersize=10,
           label=f'COLMAP ({len(colmap_centers)} cameras)',
           alpha=0.7)
    
    ax.plot(mysfm_centers[:, 0], mysfm_centers[:, 1],
           's--', color='#3498db', linewidth=3, markersize=10,
           label=f'Your SfM ({len(mysfm_centers)} cameras)',
           alpha=0.7)
    
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
    ax.set_title('Camera Trajectory Overlay Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    save_path = output_dir / 'trajectory_overlay.png'
    plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in '{output_dir}':")
    print("  ✓ 3d_comparison.png         - Side-by-side 3D reconstruction")
    print("  ✓ trajectory_comparison.png - Side-by-side trajectory (top view)")
    print("  ✓ trajectory_overlay.png    - Overlay comparison")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python visualize_cameras.py <colmap_dir> <mysfm_dir> [output_dir]")
        print("\nExample:")
        print("  python visualize_cameras.py results/colmap/sparse/0 results/my_sfm")
        print("  python visualize_cameras.py results/colmap/sparse/0 results/my_sfm comparison_vis")
        sys.exit(1)
    
    colmap_dir = sys.argv[1]
    mysfm_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'comparison_results'
    
    compare_reconstructions(colmap_dir, mysfm_dir, output_dir)