import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


class SfMModel:
    """Incremental Structure from Motion Model"""
    
    def __init__(self):
        self.cameras = {}  # cam_idx -> {'R': R, 't': t, 'K': K}
        self.points3D = {}  # pt_id -> {'xyz': [x,y,z], 'rgb': [r,g,b], 'obs': [(cam_idx, pt2d)]}
        self.next_point_id = 0
        self.registered_images = set()
        self.kp_to_pt3d = {}  # (cam_idx, kp_idx) -> pt_id
    
    def initialize_intrinsics(self, image_shape):
        """Initialize camera intrinsics from image size"""
        h, w = image_shape[:2]
        focal = max(w, h) * 1.2
        cx, cy = w / 2.0, h / 2.0
        
        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K
    
    def estimate_pose_from_essential(self, pts1, pts2, K):
        """
        Estimate relative pose between two views using Essential matrix
        
        Process:
        1. Normalize points using initial K estimate
        2. Find Essential matrix E = [t]_x R (uses 5-point algorithm internally)
        3. Decompose E to recover R and t
        
        Note: K is an initial estimate that will be refined via bundle adjustment
        
        Args:
            pts1, pts2: Matched 2D points (Nx2)
            K: Initial camera intrinsics estimate
        
        Returns:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            mask: Inlier mask (boolean array)
        """
        # Normalize points with initial K estimate
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        
        # Find Essential matrix using RANSAC
        # Note: cv2.findEssentialMat uses 5-point algorithm internally
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.001
        )
        
        if E is None:
            return None, None, None
        
        # Recover pose from Essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm)
        
        # Combine masks
        mask_essential = mask.ravel().astype(bool)
        mask_recover = mask_pose.ravel().astype(bool)
        final_mask = mask_essential & mask_recover
        
        return R, t, final_mask
    
    def triangulate_points(self, pts1, pts2, P1, P2):
        """
        Triangulate 3D points from two views
        Using DLT (Direct Linear Transform) as in lecture
        """
        # Convert to correct format for cv2.triangulatePoints
        # Input should be 2xN arrays
        pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
        pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
        
        # Triangulate using OpenCV (implements DLT)
        # cv2.triangulatePoints expects 2xN format
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        return points_3d
    
    def check_cheirality(self, points_3d, R, t):
        """Check if points are in front of both cameras"""
        # Check first camera (identity)
        depth1 = points_3d[:, 2]
        
        # Check second camera with safe computation
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            points_cam2 = (R @ points_3d.T).T + t.ravel()
            depth2 = points_cam2[:, 2]
        
        # Both depths should be positive and reasonable
        # Also check for inf/nan
        valid = (depth1 > 0) & (depth2 > 0) & \
                (depth1 < 1000) & (depth2 < 1000) & \
                np.isfinite(depth1) & np.isfinite(depth2)
        return valid
    
    def initialize_reconstruction(self, img1, img2, pts1, pts2, matches, kp1, kp2):
        """
        Initialize reconstruction from first two views
        Implements: 8-point algorithm + triangulation
        """
        print("Initializing reconstruction...")
        print(f"  Input: {len(pts1)} matched points")
        
        # Get camera intrinsics
        K = self.initialize_intrinsics(img1.shape)
        print(f"  Camera intrinsics: f={K[0,0]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        # Estimate pose
        R, t, mask = self.estimate_pose_from_essential(pts1, pts2, K)
        
        if R is None:
            print("Failed to estimate initial pose!")
            return False
        
        mask_bool = mask.ravel().astype(bool) if mask.dtype != bool else mask.ravel()
        print(f"  Pose estimation inliers: {mask_bool.sum()} / {len(mask_bool)}")
        
        # Filter by inliers
        pts1_inlier = pts1[mask_bool]
        pts2_inlier = pts2[mask_bool]
        matches_inlier = [m for i, m in enumerate(matches) if mask_bool[i]]
        
        if len(pts1_inlier) < 50:
            print(f"  Too few inliers: {len(pts1_inlier)}")
            return False
        
        # Set up projection matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        # Triangulate ONLY inlier points
        points_3d = self.triangulate_points(pts1_inlier, pts2_inlier, P1, P2)
        
        # Check cheirality
        valid_3d = self.check_cheirality(points_3d, R, t)
        print(f"  Cheirality check: {valid_3d.sum()} / {len(valid_3d)} valid")
        
        if valid_3d.sum() < 10:
            print(f"  ERROR: Only {valid_3d.sum()} valid 3D points")
            return False
        
        # Store cameras
        self.cameras[0] = {'R': np.eye(3), 't': np.zeros((3, 1)), 'K': K}
        self.cameras[1] = {'R': R, 't': t, 'K': K}
        self.registered_images = {0, 1}
        
        # Store 3D points with 2D coordinates
        for i in range(len(points_3d)):
            if valid_3d[i]:
                match = matches_inlier[i]
                kp_idx1 = match.queryIdx
                kp_idx2 = match.trainIdx
                
                # Get 2D coordinates
                pt2d_1 = pts1_inlier[i]
                pt2d_2 = pts2_inlier[i]
                
                # Get color
                x, y = int(pt2d_1[0]), int(pt2d_1[1])
                if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                    color = img1[y, x]
                    rgb = [int(color[2]), int(color[1]), int(color[0])]
                else:
                    rgb = [128, 128, 128]
                
                pt_id = self.next_point_id
                self.points3D[pt_id] = {
                    'xyz': points_3d[i],
                    'rgb': np.array(rgb),
                    # Modified: Store (cam_idx, kp_idx, pt2d) tuples
                    'obs': [
                        (0, kp_idx1, pt2d_1.copy()),
                        (1, kp_idx2, pt2d_2.copy())
                    ]
                }
                
                # Optimization: Update index
                self.kp_to_pt3d[(0, kp_idx1)] = pt_id
                self.kp_to_pt3d[(1, kp_idx2)] = pt_id
                
                self.next_point_id += 1
        
        print(f"  ✓ Initialized with {len(self.points3D)} 3D points\n")
        return True
    
    def estimate_pose_pnp(self, points_3d, points_2d, K):
        """
        Estimate camera pose using PnP with RANSAC
        Based on lecture: PnP solves for R,t given 3D-2D correspondences
        """
        # Convert to proper format
        pts_3d = np.array([p for p in points_3d], dtype=np.float64)
        pts_2d = np.array([p for p in points_2d], dtype=np.float64)
        
        if len(pts_3d) < 6:
            return None, None, None
        
        # Adaptive threshold based on image resolution
        # Your images are 640x720, so use proportional threshold
        image_diagonal = np.sqrt(640**2 + 720**2)  # ~961 pixels
        
        # Strategy 1: Start with relaxed threshold for initial registration
        if len(self.cameras) <= 3:
            # Early stage: more lenient to establish initial structure
            reproj_threshold = max(12.0, image_diagonal * 0.015)  # ~14-15 pixels
            min_inliers = 15
        else:
            # Later stage: can be more strict
            reproj_threshold = max(8.0, image_diagonal * 0.01)  # ~9-10 pixels
            min_inliers = 20
        
        # Try multiple attempts with different parameters
        best_inliers = None
        best_R = None
        best_t = None
        max_inlier_count = 0
        
        # Attempt 1: Relaxed threshold
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, None,
            iterationsCount=2000,  # Increased from 1000
            reprojectionError=reproj_threshold,
            confidence=0.999,  # Higher confidence
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if success and inliers is not None and len(inliers) >= min_inliers:
            best_inliers = inliers
            best_R, _ = cv2.Rodrigues(rvec)
            best_t = tvec
            max_inlier_count = len(inliers)
        
        # Attempt 2: If first attempt failed or got few inliers, try even more relaxed
        if max_inlier_count < len(pts_3d) * 0.3:  # Less than 30% inliers
            success2, rvec2, tvec2, inliers2 = cv2.solvePnPRansac(
                pts_3d, pts_2d, K, None,
                iterationsCount=3000,
                reprojectionError=reproj_threshold * 1.5,  # More relaxed
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE  # Try different solver
            )
            
            if success2 and inliers2 is not None and len(inliers2) > max_inlier_count:
                best_inliers = inliers2
                best_R, _ = cv2.Rodrigues(rvec2)
                best_t = tvec2
                max_inlier_count = len(inliers2)
        
        # Return best result
        if best_R is None:
            return None, None, None
        
        # Refine with inliers only (non-linear optimization)
        if max_inlier_count >= min_inliers:
            inlier_pts_3d = pts_3d[best_inliers.ravel()]
            inlier_pts_2d = pts_2d[best_inliers.ravel()]
            
            # Final refinement with iterative method
            rvec_refined, _ = cv2.Rodrigues(best_R)
            success_refine, rvec_final, tvec_final = cv2.solvePnP(
                inlier_pts_3d, inlier_pts_2d, K, None,
                rvec=rvec_refined, tvec=best_t,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success_refine:
                R_final, _ = cv2.Rodrigues(rvec_final)
                return R_final, tvec_final.reshape(3, 1), best_inliers.ravel()
        
        return best_R, best_t.reshape(3, 1), best_inliers.ravel()
        
    def register_new_image(self, img_idx, img, keypoints, matches_dict, loader):
        """
        Register new image to existing reconstruction
        Implements: PnP + RANSAC + triangulation of new points
        """
        print(f"Registering image {img_idx}...")
        
        # Optimization: Use pre-built index
        points_3d = []
        points_2d = []
        point_ids = []
        kp_indices = []
        
        for reg_idx in self.registered_images:
            i, j = min(img_idx, reg_idx), max(img_idx, reg_idx)
            if (i, j) not in matches_dict.matches:
                continue
            
            matches = matches_dict.matches[(i, j)]
            
            for m in matches:
                if img_idx < reg_idx:
                    idx_new = m.queryIdx
                    idx_reg = m.trainIdx
                else:
                    idx_new = m.trainIdx
                    idx_reg = m.queryIdx
                
                # Optimization: O(1) lookup
                key = (reg_idx, idx_reg)
                if key in self.kp_to_pt3d:
                    pt_id = self.kp_to_pt3d[key]
                    
                    if pt_id not in point_ids:  # Prevent duplicates
                        points_3d.append(self.points3D[pt_id]['xyz'])
                        pt2d = keypoints[idx_new].pt  # This is a tuple (x, y)
                        points_2d.append(pt2d)
                        point_ids.append(pt_id)
                        kp_indices.append(idx_new)
        
        if len(points_3d) < 6:
            print(f"  Not enough 3D-2D correspondences: {len(points_3d)}")
            return False
        
        print(f"  Found {len(points_3d)} 3D-2D correspondences")
        
        # Estimate pose using PnP
        K = self.cameras[0]['K']
        R, t, inliers = self.estimate_pose_pnp(points_3d, points_2d, K)
        
        if R is None:
            print("  Failed to estimate pose!")
            return False
        
        print(f"  PnP inliers: {len(inliers)} / {len(points_3d)}")
        
        # Store camera
        self.cameras[img_idx] = {'R': R, 't': t, 'K': K}
        self.registered_images.add(img_idx)
        
        # Add observations for inlier points
        for inlier_idx in inliers:
            pt_id = point_ids[inlier_idx]
            kp_idx = kp_indices[inlier_idx]
            pt2d = points_2d[inlier_idx]  # This is a tuple
            
            # Convert tuple to numpy array for consistency
            pt2d_array = np.array(pt2d, dtype=np.float64)
            
            # Add new observation to existing 3D point
            self.points3D[pt_id]['obs'].append((img_idx, kp_idx, pt2d_array))
            
            # Update index
            self.kp_to_pt3d[(img_idx, kp_idx)] = pt_id
        
        # Triangulate new points
        self._triangulate_new_points(img_idx, img, keypoints, matches_dict, loader)
        
        print(f"  Total 3D points: {len(self.points3D)}\n")
        return True
    
    def _triangulate_new_points(self, new_idx, img, keypoints, matches_dict, loader):
        """Triangulate new 3D points from newly registered image"""
        
        n_existing_points = len(self.points3D)
        n_cameras = len(self.cameras)
        
        # Adaptive filtering based on reconstruction maturity
        if n_cameras <= 3:
            # Very early stage: Be very lenient
            min_matches = 5
            depth_threshold = 2000
            reproj_threshold = 20.0  # Very lenient
        elif n_cameras <= 6:
            # Early stage: Still lenient
            min_matches = 6
            depth_threshold = 1500
            reproj_threshold = 15.0  # Lenient
        elif n_existing_points < 3000:
            # Mid stage: Moderate
            min_matches = 7
            depth_threshold = 1200
            reproj_threshold = 12.0  # Moderate
        else:
            # Late stage: More strict
            min_matches = 8
            depth_threshold = 1000
            reproj_threshold = 10.0  # Strict
        
        K = self.cameras[new_idx]['K']
        R_new = self.cameras[new_idx]['R']
        t_new = self.cameras[new_idx]['t']
        P_new = K @ np.hstack([R_new, t_new])
        
        added_count = 0
        
        for reg_idx in list(self.registered_images):
            if reg_idx == new_idx:
                continue
            
            i, j = min(new_idx, reg_idx), max(new_idx, reg_idx)
            if (i, j) not in matches_dict.matches:
                continue
            
            matches = matches_dict.matches[(i, j)]
            
            R_reg = self.cameras[reg_idx]['R']
            t_reg = self.cameras[reg_idx]['t']
            P_reg = K @ np.hstack([R_reg, t_reg])
            
            pts_new = []
            pts_reg = []
            kp_idx_new = []
            kp_idx_reg = []
            
            for m in matches:
                if new_idx < reg_idx:
                    idx_new = m.queryIdx
                    idx_reg = m.trainIdx
                else:
                    idx_new = m.trainIdx
                    idx_reg = m.queryIdx
                
                if (new_idx, idx_new) in self.kp_to_pt3d or (reg_idx, idx_reg) in self.kp_to_pt3d:
                    continue
                
                pt_new = keypoints[idx_new].pt
                pt_reg = loader.keypoints[reg_idx][idx_reg].pt
                
                pts_new.append(pt_new)
                pts_reg.append(pt_reg)
                kp_idx_new.append(idx_new)
                kp_idx_reg.append(idx_reg)
            
            if len(pts_new) < min_matches:
                continue
            
            pts_new = np.array(pts_new, dtype=np.float32)
            pts_reg = np.array(pts_reg, dtype=np.float32)
            
            points_3d = self.triangulate_points(pts_new, pts_reg, P_new, P_reg)
            
            # Multi-stage validation
            valid_indices = []
            
            # Pre-compute rotation vectors (once per registered camera pair)
            rvec_new, _ = cv2.Rodrigues(R_new)
            rvec_reg, _ = cv2.Rodrigues(R_reg)
            
            for idx in range(len(points_3d)):
                pt3d = points_3d[idx]
                
                # Stage 1: Basic depth check
                depth_new = pt3d[2]
                if depth_new <= 0 or depth_new > depth_threshold:
                    continue
                
                # Stage 2: Check depth in registered camera
                pt_cam_reg = R_reg @ pt3d.reshape(3, 1) + t_reg
                depth_reg = pt_cam_reg[2, 0]
                if depth_reg <= 0 or depth_reg > depth_threshold:
                    continue
                
                # Stage 3: Reprojection error check (adaptive threshold)
                pt2d_new_obs = pts_new[idx]
                pt2d_reg_obs = pts_reg[idx]
                
                pt2d_new_proj = self._project_point(pt3d, rvec_new, t_new.ravel(), K)
                pt2d_reg_proj = self._project_point(pt3d, rvec_reg, t_reg.ravel(), K)
                
                error_new = np.linalg.norm(pt2d_new_obs - pt2d_new_proj)
                error_reg = np.linalg.norm(pt2d_reg_obs - pt2d_reg_proj)
                
                # Use adaptive threshold
                if error_new < reproj_threshold and error_reg < reproj_threshold:
                    valid_indices.append(idx)
            
            # Add valid points
            for idx in valid_indices:
                if n_existing_points + added_count > 10000:  # Increased limit
                    print(f"  Point limit reached ({n_existing_points + added_count}), stopping triangulation")
                    return
                
                pt3d = points_3d[idx]
                pt2d_new = pts_new[idx]
                pt2d_reg = pts_reg[idx]
                
                x, y = int(pt2d_new[0]), int(pt2d_new[1])
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    color = img[y, x]
                    rgb = [int(color[2]), int(color[1]), int(color[0])]
                else:
                    rgb = [128, 128, 128]
                
                pt_id = self.next_point_id
                self.points3D[pt_id] = {
                    'xyz': pt3d,
                    'rgb': np.array(rgb),
                    'obs': [
                        (new_idx, kp_idx_new[idx], pt2d_new.copy()),
                        (reg_idx, kp_idx_reg[idx], pt2d_reg.copy())
                    ]
                }
                
                self.kp_to_pt3d[(new_idx, kp_idx_new[idx])] = pt_id
                self.kp_to_pt3d[(reg_idx, kp_idx_reg[idx])] = pt_id
                
                self.next_point_id += 1
                added_count += 1
        
    def bundle_adjustment(self, optimize_intrinsics=True):
        """
        Bundle Adjustment: Jointly optimize camera poses, 3D points, and intrinsics
        
        Minimizes reprojection error: Σ ||x_observed - proj(K, R, t, X)||²
        
        Based on lecture: Non-linear refinement of structure and motion
        
        Args:
            optimize_intrinsics: If True, also optimize focal length and principal point
        """
        print("Running bundle adjustment...")
        
        if len(self.cameras) < 2 or len(self.points3D) < 10:
            print("  Skipping: not enough data\n")
            return
        
        n_cameras = len(self.cameras)
        n_points = len(self.points3D)
        cam_indices = sorted(self.cameras.keys())
        point_indices = list(self.points3D.keys())
        
        print(f"  Cameras: {n_cameras}, Points: {n_points}")
        print(f"  Optimize intrinsics: {optimize_intrinsics}")
        
        # Create index mappings
        cam_idx_map = {idx: i for i, idx in enumerate(cam_indices)}
        pt_idx_map = {idx: i for i, idx in enumerate(point_indices)}
        
        # ========== Prepare initial parameters ==========
        K = self.cameras[cam_indices[0]]['K']
        
        if optimize_intrinsics:
            # Intrinsic parameters: [focal, cx, cy]
            # Assume square pixels (fx = fy) and zero skew
            intrinsic_params = [K[0, 0], K[0, 2], K[1, 2]]
        else:
            intrinsic_params = []
        
        # Camera extrinsics: [rvec (3), tvec (3)] for each camera
        camera_params = []
        for cam_idx in cam_indices:
            cam = self.cameras[cam_idx]
            rvec, _ = cv2.Rodrigues(cam['R'])
            camera_params.extend(rvec.ravel())
            camera_params.extend(cam['t'].ravel())
        
        # 3D point positions: [x, y, z] for each point
        point_params = []
        for pt_id in point_indices:
            point_params.extend(self.points3D[pt_id]['xyz'])
        
        # Combine all parameters into single vector
        x0 = np.hstack([intrinsic_params, camera_params, point_params])
        
        print(f"  Total parameters: {len(x0)}")
        print(f"    - Intrinsics: {len(intrinsic_params)}")
        print(f"    - Camera params: {len(camera_params)}")
        print(f"    - Point params: {len(point_params)}")
        
        # ========== Collect observations ==========
        observations = []
        for pt_id in point_indices:
            pt_data = self.points3D[pt_id]
            pt_idx = pt_idx_map[pt_id]
            
            for cam_idx, kp_idx, pt2d_obs in pt_data['obs']:
                if cam_idx not in cam_idx_map:
                    continue
                
                observations.append({
                    'cam_idx': cam_idx_map[cam_idx],
                    'pt_idx': pt_idx,
                    'pt2d': np.array(pt2d_obs, dtype=np.float64)
                })
        
        n_obs = len(observations)
        print(f"  Observations: {n_obs}")
        
        if n_obs < 20:
            print("  Skipping: too few observations\n")
            return
        
        # Convert observations to arrays for vectorization
        obs_cam_indices = np.array([obs['cam_idx'] for obs in observations], dtype=np.int32)
        obs_pt_indices = np.array([obs['pt_idx'] for obs in observations], dtype=np.int32)
        obs_pt2d = np.array([obs['pt2d'] for obs in observations], dtype=np.float64)
        
        # ========== Define residual function ==========
        def residuals(params):
            """
            Compute reprojection errors for all observations
            
            Returns:
                residuals: 2D array of shape (n_obs * 2,) containing x and y errors
            """
            # Parse parameters
            if optimize_intrinsics:
                focal = params[0]
                cx = params[1]
                cy = params[2]
                cam_start = 3
            else:
                focal = K[0, 0]
                cx = K[0, 2]
                cy = K[1, 2]
                cam_start = 0
            
            n_cam_params = n_cameras * 6
            cam_params = params[cam_start:cam_start + n_cam_params].reshape(n_cameras, 6)
            pt_params = params[cam_start + n_cam_params:].reshape(n_points, 3)
            
            rvecs = cam_params[:, :3]
            tvecs = cam_params[:, 3:6]
            
            # Precompute rotation matrices
            R_matrices = np.zeros((n_cameras, 3, 3))
            for i in range(n_cameras):
                R_matrices[i], _ = cv2.Rodrigues(rvecs[i])
            
            # Vectorized projection
            cam_ids = obs_cam_indices
            pt_ids = obs_pt_indices
            
            Rs = R_matrices[cam_ids]  # (n_obs, 3, 3)
            ts = tvecs[cam_ids]       # (n_obs, 3)
            pts3d = pt_params[pt_ids] # (n_obs, 3)
            
            # Transform to camera coordinates: p_cam = R @ p_world + t
            pts_cam = np.einsum('nij,nj->ni', Rs, pts3d) + ts
            
            # Handle points behind camera
            behind_camera = pts_cam[:, 2] <= 0
            pts_cam[behind_camera, 2] = 1e-6  # Avoid division by zero
            
            # Perspective projection
            x_norm = pts_cam[:, 0] / pts_cam[:, 2]
            y_norm = pts_cam[:, 1] / pts_cam[:, 2]
            
            # Apply intrinsics
            u_proj = focal * x_norm + cx
            v_proj = focal * y_norm + cy
            
            pts2d_proj = np.column_stack([u_proj, v_proj])
            
            # Compute residuals
            residuals = (obs_pt2d - pts2d_proj).ravel()
            
            # Penalize points behind camera
            residuals[np.repeat(behind_camera, 2)] = 1000.0
            
            return residuals
        
        # ========== Run optimization ==========
        print(f"  Running optimization...")
        import time
        start_time = time.time()
        
        result = least_squares(
            residuals,
            x0,
            method='trf',
            ftol=1e-4,
            xtol=1e-4,
            max_nfev=100,
            verbose=0
        )
        
        elapsed = time.time() - start_time
        
        # ========== Report results ==========
        initial_cost = np.sum(residuals(x0)**2)
        final_cost = np.sum(result.fun**2)
        improvement = (1 - final_cost/initial_cost) * 100 if initial_cost > 0 else 0
        mean_reproj_error = np.sqrt(final_cost / n_obs)
        
        print(f"  Initial cost: {initial_cost:.2f}")
        print(f"  Final cost: {final_cost:.2f} ({improvement:.1f}% improvement)")
        print(f"  Mean reprojection error: {mean_reproj_error:.3f} pixels")
        print(f"  Iterations: {result.nfev}")
        print(f"  Time: {elapsed:.2f}s")
        
        # ========== Update model with optimized parameters ==========
        if optimize_intrinsics:
            focal_opt = result.x[0]
            cx_opt = result.x[1]
            cy_opt = result.x[2]
            
            K_opt = np.array([
                [focal_opt, 0, cx_opt],
                [0, focal_opt, cy_opt],
                [0, 0, 1]
            ], dtype=np.float64)
            
            print(f"  Intrinsics refinement:")
            print(f"    focal: {K[0,0]:.1f} → {focal_opt:.1f}")
            print(f"    cx: {K[0,2]:.1f} → {cx_opt:.1f}")
            print(f"    cy: {K[1,2]:.1f} → {cy_opt:.1f}")
            
            # Update all cameras with refined intrinsics
            for cam_idx in cam_indices:
                self.cameras[cam_idx]['K'] = K_opt.copy()
            
            cam_start = 3
        else:
            cam_start = 0
        
        # Update camera extrinsics
        n_cam_params = n_cameras * 6
        optimized_cam_params = result.x[cam_start:cam_start + n_cam_params].reshape(n_cameras, 6)
        
        for i, cam_idx in enumerate(cam_indices):
            rvec = optimized_cam_params[i, :3]
            tvec = optimized_cam_params[i, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            self.cameras[cam_idx]['R'] = R
            self.cameras[cam_idx]['t'] = tvec.reshape(3, 1)
        
        # Update 3D points
        optimized_pt_params = result.x[cam_start + n_cam_params:].reshape(n_points, 3)
        for i, pt_id in enumerate(point_indices):
            self.points3D[pt_id]['xyz'] = optimized_pt_params[i]
        
        print("  Bundle adjustment completed\n")
            
    def _project_point(self, pt3d, rvec, tvec, K):
        """
        Project 3D point to 2D image plane
        Based on lecture: x = K[R|t]X
        
        Args:
            pt3d: 3D point in world coordinates (3,)
            rvec: Rotation vector (3,)
            tvec: Translation vector (3,)
            K: Camera intrinsics matrix (3, 3)
        
        Returns:
            pt2d: Projected 2D point (2,)
        """
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Transform point to camera coordinates
        pt_cam = R @ pt3d.reshape(3, 1) + tvec.reshape(3, 1)
        
        # Project to image plane
        pt_img = K @ pt_cam
        
        # Normalize by depth
        pt2d = (pt_img[:2] / pt_img[2]).ravel()
        
        return pt2d
    
    def save_results(self, output_dir):
        """Save reconstruction results in COLMAP format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cameras.txt
        with open(output_dir / 'cameras.txt', 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            if self.cameras:
                K = self.cameras[0]['K']
                f.write(f"1 OPENCV 640 480 {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]} 0 0 0 0\n")
        
        # Save images.txt
        with open(output_dir / 'images.txt', 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for img_idx in sorted(self.cameras.keys()):
                cam = self.cameras[img_idx]
                R = cam['R']
                t = cam['t'].ravel()
                
                # Convert rotation matrix to quaternion
                rvec, _ = cv2.Rodrigues(R)
                angle = np.linalg.norm(rvec)
                if angle > 0:
                    axis = rvec.ravel() / angle
                    qw = np.cos(angle / 2)
                    qx, qy, qz = np.sin(angle / 2) * axis
                else:
                    qw, qx, qy, qz = 1, 0, 0, 0
                
                f.write(f"{img_idx+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 image_{img_idx:04d}.jpg\n")
                f.write("\n")
        
        # Save points3D.txt
        with open(output_dir / 'points3D.txt', 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            for pt_id, pt_data in self.points3D.items():
                xyz = pt_data['xyz']
                rgb = pt_data['rgb']
                f.write(f"{pt_id} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} 0.0")
                
                # Modified: obs now has (cam_idx, kp_idx, pt2d) format
                for cam_idx, kp_idx, pt2d in pt_data['obs']:
                    f.write(f" {cam_idx+1} 0")
                f.write("\n")
        
        # Save vis.txt (visualization format)
        with open(output_dir / 'vis.txt', 'w') as f:
            for pt_data in self.points3D.values():
                xyz = pt_data['xyz']
                rgb = pt_data['rgb']
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
        
        print(f"Results saved to {output_dir}/")
    
    def visualize(self, output_dir):
        """Create visualization matching COLMAP example style"""
        output_dir = Path(output_dir)
        
        # Check if vis.txt exists and has data
        vis_file = output_dir / 'vis.txt'
        if not vis_file.exists() or vis_file.stat().st_size == 0:
            print("No 3D points to visualize - skipping visualization")
            return
        
        # Load vis.txt
        try:
            data = np.loadtxt(vis_file)
            if data.ndim == 1 or len(data) == 0:
                print("Not enough 3D points to visualize - skipping visualization")
                return
        except Exception as e:
            print(f"Error loading visualization data: {e}")
            return
        
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor('#2d3561')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2d3561')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=1, alpha=0.6)
        
        # Add reference circles
        center = np.mean(points, axis=0)
        theta = np.linspace(0, 2*np.pi, 100)
        
        for i, j, k in [(0,1,2), (0,2,1), (1,2,0)]:
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
        plt.savefig(output_dir / 'visualization.png', dpi=300,
                   bbox_inches='tight', facecolor='#2d3561')
        plt.close()
        
        print(f"Visualization saved to {output_dir}/visualization.png")
