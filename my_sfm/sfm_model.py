import cv2
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


class SfMModel:
    """
    Incremental Structure from Motion - Following Lecture 8
    
    Pipeline:
    1. Initialize from two views (Lecture 5: Essential Matrix)
    2. Iteratively add new cameras (Lecture 4: PnP)
    3. Triangulate new points (Lecture 6: DLT Triangulation)
    4. Bundle Adjustment (Lecture 8: Non-linear refinement)
    """
    
    def __init__(self):
        self.cameras = {}
        self.points3D = {}
        self.next_point_id = 0
        self.registered_images = set()
        self.kp_to_pt3d = {}
    
    def initialize_intrinsics(self, image_shape):
        """Initialize camera intrinsics - Lecture 4"""
        h, w = image_shape[:2]
        focal = max(w, h) * 1.2
        cx, cy = w / 2.0, h / 2.0
        
        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K
    
    # ============================================================================
    # 1. INITIAL POSE ESTIMATION (Lecture 5: Epipolar Geometry)
    # ============================================================================
    
    def estimate_pose_from_essential(self, pts1, pts2, K):
        """
        Lecture 5: Essential Matrix E = [t]_x R
        5-point algorithm + decomposition
        
        Returns R, t with UNIT SCALE (scale ambiguity)
        """
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.001
        )
        
        if E is None:
            return None, None, None
        
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm)
        
        mask_essential = mask.ravel().astype(bool)
        mask_recover = mask_pose.ravel().astype(bool)
        final_mask = mask_essential & mask_recover
        
        return R, t, final_mask
    
    def triangulate_points_dlt(self, pts1, pts2, P1, P2):
        pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
        pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
        n = len(pts1)
        
        points_3d = []
        
        for i in range(n):
            # DLT: Construct A matrix for each point
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]
            
            A = np.array([
                x1 * P1[2] - P1[0],
                y1 * P1[2] - P1[1],
                x2 * P2[2] - P2[0],
                y2 * P2[2] - P2[1]
            ])
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]  # Normalize homogeneous coordinates
            
            points_3d.append(X[:3])
        
        return np.array(points_3d)
    
    def check_cheirality(self, points_3d, R, t):
        """Check if points are in front of both cameras"""
        depth1 = points_3d[:, 2]
        
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            points_cam2 = (R @ points_3d.T).T + t.ravel()
            depth2 = points_cam2[:, 2]
        
        valid = (depth1 > 0) & (depth2 > 0) & \
                (depth1 < 1000) & (depth2 < 1000) & \
                np.isfinite(depth1) & np.isfinite(depth2)
        return valid
    
    def normalize_scale(self, points_3d, target_median_depth=10.0):
        """
        FIX: Normalize scale to consistent metric
        Essential matrix gives unit translation - need to fix scale
        """
        depths = points_3d[:, 2]
        valid_depths = depths[(depths > 0) & (depths < 1000)]
        
        if len(valid_depths) == 0:
            return points_3d, 1.0
        
        current_median = np.median(valid_depths)
        scale_factor = target_median_depth / current_median
        
        return points_3d * scale_factor, scale_factor
    
    def initialize_reconstruction(self, img1, img2, pts1, pts2, matches, kp1, kp2):
        """
        Lecture 8: Initialize SfM from two views
        
        Steps:
        1. Estimate Essential Matrix (5-point algorithm)
        2. Recover R, t (4 solutions -> test cheirality)
        3. Triangulate initial points
        4. CRITICAL: Normalize scale for consistency
        """
        print("Initializing reconstruction...")
        print(f"  Input: {len(pts1)} matched points")
        
        K = self.initialize_intrinsics(img1.shape)
        print(f"  Camera intrinsics: f={K[0,0]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        # Step 1-2: Essential Matrix + Pose Recovery
        R, t, mask = self.estimate_pose_from_essential(pts1, pts2, K)
        
        if R is None:
            print("Failed to estimate initial pose!")
            return False
        
        mask_bool = mask.ravel().astype(bool) if mask.dtype != bool else mask.ravel()
        print(f"  Pose estimation inliers: {mask_bool.sum()} / {len(mask_bool)}")
        
        pts1_inlier = pts1[mask_bool]
        pts2_inlier = pts2[mask_bool]
        matches_inlier = [m for i, m in enumerate(matches) if mask_bool[i]]
        
        if len(pts1_inlier) < 50:
            print(f"  Too few inliers: {len(pts1_inlier)}")
            return False
        
        # Step 3: Triangulate
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        points_3d = self.triangulate_points_dlt(pts1_inlier, pts2_inlier, P1, P2)
        
        valid_3d = self.check_cheirality(points_3d, R, t)
        print(f"  Cheirality check: {valid_3d.sum()} / {len(valid_3d)} valid")
        
        if valid_3d.sum() < 10:
            print(f"  ERROR: Only {valid_3d.sum()} valid 3D points")
            return False
        
        # CRITICAL FIX: Normalize scale
        points_3d_valid = points_3d[valid_3d]
        points_3d_normalized, scale_factor = self.normalize_scale(points_3d_valid)
        
        # Scale translation accordingly
        t_scaled = t * scale_factor
        
        print(f"  Scale normalization: factor={scale_factor:.3f}")
        print(f"  Depth range: [{points_3d_normalized[:, 2].min():.2f}, {points_3d_normalized[:, 2].max():.2f}]")
        
        # Store cameras
        self.cameras[0] = {'R': np.eye(3), 't': np.zeros((3, 1)), 'K': K}
        self.cameras[1] = {'R': R, 't': t_scaled, 'K': K}
        self.registered_images = {0, 1}
        
        # Store 3D points
        valid_idx = 0
        for i in range(len(points_3d)):
            if not valid_3d[i]:
                continue
                
            match = matches_inlier[i]
            kp_idx1 = match.queryIdx
            kp_idx2 = match.trainIdx
            
            pt2d_1 = pts1_inlier[i]
            pt2d_2 = pts2_inlier[i]
            
            x, y = int(pt2d_1[0]), int(pt2d_1[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                color = img1[y, x]
                rgb = [int(color[2]), int(color[1]), int(color[0])]
            else:
                rgb = [128, 128, 128]
            
            pt_id = self.next_point_id
            self.points3D[pt_id] = {
                'xyz': points_3d_normalized[valid_idx],
                'rgb': np.array(rgb),
                'obs': [
                    (0, kp_idx1, pt2d_1.copy()),
                    (1, kp_idx2, pt2d_2.copy())
                ]
            }
            
            self.kp_to_pt3d[(0, kp_idx1)] = pt_id
            self.kp_to_pt3d[(1, kp_idx2)] = pt_id
            
            self.next_point_id += 1
            valid_idx += 1
        
        print(f"  ✓ Initialized with {len(self.points3D)} 3D points\n")
        return True
    
    # ============================================================================
    # 2. INCREMENTAL CAMERA REGISTRATION (Lecture 4: PnP)
    # ============================================================================
    
    def estimate_pose_pnp_dlt(self, points_3d, points_2d, K):
        """
        Lecture 4: PnP using DLT
        
        Solve for projection matrix P = K[R|t], then decompose
        Must implement ourselves (cv2.solvePnP not allowed for registration)
        """
        from scipy.optimize import least_squares
        
        pts_3d = np.array(points_3d, dtype=np.float64)
        pts_2d = np.array(points_2d, dtype=np.float64)
        n = len(pts_3d)
        
        if n < 6:
            return None, None, None
        
        # Initial guess from existing cameras
        if len(self.cameras) >= 2:
            all_ts = [cam['t'].ravel() for cam in self.cameras.values()]
            all_Rs = [cam['R'] for cam in self.cameras.values()]
            
            # Median camera position
            median_t = np.median(all_ts, axis=0)
            
            # Average rotation
            rvecs = [cv2.Rodrigues(R)[0].ravel() for R in all_Rs]
            median_rvec = np.median(rvecs, axis=0)
            R_init, _ = cv2.Rodrigues(median_rvec)
            
            rvec0 = median_rvec
            tvec0 = median_t
        else:
            rvec0 = np.zeros(3)
            tvec0 = np.array([0, 0, 10])
        
        x0 = np.hstack([rvec0, tvec0])
        
        # Reprojection error
        def residuals(x):
            rvec = x[:3]
            tvec = x[3:6]
            
            try:
                R, _ = cv2.Rodrigues(rvec)
                
                # Project: x = K[R|t]X
                pts_cam = (R @ pts_3d.T).T + tvec
                
                # Filter behind camera
                valid = pts_cam[:, 2] > 0.1
                
                errors = np.zeros(n * 2)
                for i in range(n):
                    if not valid[i]:
                        errors[2*i:2*i+2] = 1000.0
                        continue
                    
                    pt_proj = K @ pts_cam[i]
                    u = pt_proj[0] / pt_proj[2]
                    v = pt_proj[1] / pt_proj[2]
                    
                    errors[2*i] = pts_2d[i, 0] - u
                    errors[2*i+1] = pts_2d[i, 1] - v
                
                return errors
            except:
                return np.ones(n * 2) * 1000.0
        
        # Optimize
        try:
            result = least_squares(
                residuals, x0,
                method='trf',
                ftol=1e-4,
                xtol=1e-4,
                max_nfev=50,
                verbose=0
            )
            
            rvec_opt = result.x[:3]
            tvec_opt = result.x[3:6]
            
            R, _ = cv2.Rodrigues(rvec_opt)
            t = tvec_opt.reshape(3, 1)
            
            # Compute errors
            errors = residuals(result.x).reshape(-1, 2)
            errors = np.linalg.norm(errors, axis=1)
            errors[errors > 500] = 1e6
            
            return R, t, errors
        except:
            return None, None, None
    
    def estimate_pose_pnp_ransac(self, points_3d, points_2d, K,
                                  max_iterations=300):
        """
        RANSAC wrapper for PnP (Lecture 8)
        """
        pts_3d = np.array(points_3d, dtype=np.float64)
        pts_2d = np.array(points_2d, dtype=np.float64)
        n = len(pts_3d)
        
        if n < 6:
            return None, None, None
        
        # Adaptive thresholds based on reconstruction maturity
        if len(self.cameras) <= 3:
            threshold = 15.0  # More lenient early on
            min_inliers = 15
            sample_size = 8
        else:
            threshold = 10.0
            min_inliers = 20
            sample_size = 6
        
        print(f"  Running PnP-RANSAC (DLT-based)...")
        print(f"    Points: {n}, Threshold: {threshold:.1f}px, Sample: {sample_size}")
        
        best_R = None
        best_t = None
        best_inliers = None
        max_inlier_count = 0
        
        for i in range(max_iterations):
            # Sample
            indices = np.random.choice(n, min(sample_size, n), replace=False)
            sample_3d = pts_3d[indices]
            sample_2d = pts_2d[indices]
            
            # Estimate
            R, t, _ = self.estimate_pose_pnp_dlt(sample_3d, sample_2d, K)
            
            if R is None:
                continue
            
            # Test all
            _, _, errors = self.estimate_pose_pnp_dlt(pts_3d, pts_2d, K)
            
            if errors is None:
                continue
            
            # Count inliers
            inliers = errors < threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_R = R
                best_t = t
                best_inliers = inliers
            
            # Early stop
            if inlier_count > n * 0.5:
                break
        
        if best_R is None or max_inlier_count < min_inliers:
            print(f"    Failed: {max_inlier_count}/{n} inliers (need {min_inliers})")
            return None, None, None
        
        # Refine with all inliers
        inlier_pts_3d = pts_3d[best_inliers]
        inlier_pts_2d = pts_2d[best_inliers]
        
        R_refined, t_refined, _ = self.estimate_pose_pnp_dlt(
            inlier_pts_3d, inlier_pts_2d, K
        )
        
        if R_refined is not None:
            best_R = R_refined
            best_t = t_refined
            
            # Recompute inliers
            _, _, errors_final = self.estimate_pose_pnp_dlt(pts_3d, pts_2d, K)
            if errors_final is not None:
                best_inliers = errors_final < threshold
                max_inlier_count = np.sum(best_inliers)
        
        inlier_ratio = max_inlier_count / n * 100
        print(f"    Success: {max_inlier_count}/{n} inliers ({inlier_ratio:.1f}%)")
        
        return best_R, best_t, np.where(best_inliers)[0]
    
    def register_new_image(self, img_idx, img, keypoints, matches_dict, loader):
        """
        Lecture 8: Add new camera to reconstruction
        """
        print(f"Registering image {img_idx}...")
        
        # Collect 3D-2D correspondences
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
                
                key = (reg_idx, idx_reg)
                if key in self.kp_to_pt3d:
                    pt_id = self.kp_to_pt3d[key]
                    
                    if pt_id not in point_ids:
                        points_3d.append(self.points3D[pt_id]['xyz'])
                        pt2d = keypoints[idx_new].pt
                        points_2d.append(pt2d)
                        point_ids.append(pt_id)
                        kp_indices.append(idx_new)
        
        if len(points_3d) < 6:
            print(f"  Not enough correspondences: {len(points_3d)}")
            return False
        
        print(f"  Found {len(points_3d)} 3D-2D correspondences")
        
        # PnP-RANSAC
        K = self.cameras[0]['K']
        R, t, inliers = self.estimate_pose_pnp_ransac(points_3d, points_2d, K)
        
        if R is None:
            print("  Failed to estimate pose!")
            return False
        
        print(f"  PnP inliers: {len(inliers)} / {len(points_3d)}")
        
        # Store camera
        self.cameras[img_idx] = {'R': R, 't': t, 'K': K}
        self.registered_images.add(img_idx)
        
        # Add observations
        for inlier_idx in inliers:
            pt_id = point_ids[inlier_idx]
            kp_idx = kp_indices[inlier_idx]
            pt2d = points_2d[inlier_idx]
            
            pt2d_array = np.array(pt2d, dtype=np.float64)
            
            self.points3D[pt_id]['obs'].append((img_idx, kp_idx, pt2d_array))
            self.kp_to_pt3d[(img_idx, kp_idx)] = pt_id
        
        # Triangulate new points
        self._triangulate_new_points(img_idx, img, keypoints, matches_dict, loader)
        
        print(f"  Total 3D points: {len(self.points3D)}\n")
        return True
    
    # ============================================================================
    # 3. TRIANGULATION OF NEW POINTS (Lecture 6)
    # ============================================================================
    
    def _triangulate_new_points(self, new_idx, img, keypoints, matches_dict, loader):
        """Lecture 6: Triangulate new 3D points"""
        
        K = self.cameras[new_idx]['K']
        R_new = self.cameras[new_idx]['R']
        t_new = self.cameras[new_idx]['t']
        P_new = K @ np.hstack([R_new, t_new])
        
        added_count = 0
        reproj_threshold = 8.0
        
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
                
                # Skip if already triangulated
                if (new_idx, idx_new) in self.kp_to_pt3d or (reg_idx, idx_reg) in self.kp_to_pt3d:
                    continue
                
                pt_new = keypoints[idx_new].pt
                pt_reg = loader.keypoints[reg_idx][idx_reg].pt
                
                pts_new.append(pt_new)
                pts_reg.append(pt_reg)
                kp_idx_new.append(idx_new)
                kp_idx_reg.append(idx_reg)
            
            if len(pts_new) < 5:
                continue
            
            pts_new = np.array(pts_new, dtype=np.float32)
            pts_reg = np.array(pts_reg, dtype=np.float32)
            
            # Triangulate
            points_3d = self.triangulate_points_dlt(pts_new, pts_reg, P_new, P_reg)
            
            # Validate
            for idx in range(len(points_3d)):
                pt3d = points_3d[idx]
                
                # Depth check
                if pt3d[2] <= 0 or pt3d[2] > 1000:
                    continue
                
                pt_cam_reg = R_reg @ pt3d + t_reg.ravel()
                if pt_cam_reg[2] <= 0 or pt_cam_reg[2] > 1000:
                    continue
                
                # Reprojection check
                pt_cam_new = R_new @ pt3d + t_new.ravel()
                pt_img_new = K @ pt_cam_new
                pt2d_new_proj = pt_img_new[:2] / pt_img_new[2]
                
                pt_img_reg = K @ pt_cam_reg
                pt2d_reg_proj = pt_img_reg[:2] / pt_img_reg[2]
                
                error_new = np.linalg.norm(pts_new[idx] - pt2d_new_proj)
                error_reg = np.linalg.norm(pts_reg[idx] - pt2d_reg_proj)
                
                if error_new < reproj_threshold and error_reg < reproj_threshold:
                    # Add point
                    x, y = int(pts_new[idx][0]), int(pts_new[idx][1])
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
                            (new_idx, kp_idx_new[idx], pts_new[idx].copy()),
                            (reg_idx, kp_idx_reg[idx], pts_reg[idx].copy())
                        ]
                    }
                    
                    self.kp_to_pt3d[(new_idx, kp_idx_new[idx])] = pt_id
                    self.kp_to_pt3d[(reg_idx, kp_idx_reg[idx])] = pt_id
                    
                    self.next_point_id += 1
                    added_count += 1
                    
                    if added_count > 500:  # Limit per pair
                        break
            
            if added_count > 500:
                break
    
    # ============================================================================
    # 4. BUNDLE ADJUSTMENT (Lecture 8: Non-linear refinement)
    # ============================================================================
    
    def bundle_adjustment(self, optimize_intrinsics=True):
        """
        Lecture 8: Bundle Adjustment
        Minimize: Σ ||x_ij - proj(X_j, P_i)||²
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
        
        # Mappings
        cam_idx_map = {idx: i for i, idx in enumerate(cam_indices)}
        pt_idx_map = {idx: i for i, idx in enumerate(point_indices)}
        
        # Parameters
        K = self.cameras[cam_indices[0]]['K']
        
        if optimize_intrinsics:
            intrinsic_params = [K[0, 0], K[0, 2], K[1, 2]]
        else:
            intrinsic_params = []
        
        camera_params = []
        for cam_idx in cam_indices:
            cam = self.cameras[cam_idx]
            rvec, _ = cv2.Rodrigues(cam['R'])
            camera_params.extend(rvec.ravel())
            camera_params.extend(cam['t'].ravel())
        
        point_params = []
        for pt_id in point_indices:
            point_params.extend(self.points3D[pt_id]['xyz'])
        
        x0 = np.hstack([intrinsic_params, camera_params, point_params])
        
        # Observations
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
        
        # Residual function
        def residuals(params):
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
            
            residuals_list = []
            
            for obs in observations:
                cam_id = obs['cam_idx']
                pt_id = obs['pt_idx']
                pt2d_obs = obs['pt2d']
                
                rvec = cam_params[cam_id, :3]
                tvec = cam_params[cam_id, 3:6]
                pt3d = pt_params[pt_id]
                
                # Project
                R, _ = cv2.Rodrigues(rvec)
                pt_cam = R @ pt3d + tvec
                
                if pt_cam[2] <= 0:
                    residuals_list.extend([1000.0, 1000.0])
                    continue
                
                u = focal * pt_cam[0] / pt_cam[2] + cx
                v = focal * pt_cam[1] / pt_cam[2] + cy
                
                residuals_list.append(pt2d_obs[0] - u)
                residuals_list.append(pt2d_obs[1] - v)
            
            return np.array(residuals_list)
        
        # Optimize
        print(f"  Running optimization...")
        import time
        start_time = time.time()
        
        result = least_squares(
            residuals,
            x0,
            method='trf',
            ftol=1e-4,
            xtol=1e-4,
            max_nfev=30,  # Limited for speed
            verbose=0
        )
        
        elapsed = time.time() - start_time
        
        # Report
        initial_cost = np.sum(residuals(x0)**2)
        final_cost = np.sum(result.fun**2)
        improvement = (1 - final_cost/initial_cost) * 100 if initial_cost > 0 else 0
        mean_reproj_error = np.sqrt(final_cost / n_obs)
        
        print(f"  Initial cost: {initial_cost:.2f}")
        print(f"  Final cost: {final_cost:.2f} ({improvement:.1f}% improvement)")
        print(f"  Mean reprojection error: {mean_reproj_error:.3f} pixels")
        print(f"  Time: {elapsed:.2f}s")
        
        # Update model
        if optimize_intrinsics:
            focal_opt = result.x[0]
            cx_opt = result.x[1]
            cy_opt = result.x[2]
            
            K_opt = np.array([
                [focal_opt, 0, cx_opt],
                [0, focal_opt, cy_opt],
                [0, 0, 1]
            ])
            
            print(f"  Intrinsics: f={K[0,0]:.1f}→{focal_opt:.1f}")
            
            for cam_idx in cam_indices:
                self.cameras[cam_idx]['K'] = K_opt.copy()
            
            cam_start = 3
        else:
            cam_start = 0
        
        # Update cameras
        n_cam_params = n_cameras * 6
        optimized_cam_params = result.x[cam_start:cam_start + n_cam_params].reshape(n_cameras, 6)
        
        for i, cam_idx in enumerate(cam_indices):
            rvec = optimized_cam_params[i, :3]
            tvec = optimized_cam_params[i, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            self.cameras[cam_idx]['R'] = R
            self.cameras[cam_idx]['t'] = tvec.reshape(3, 1)
        
        # Update points
        optimized_pt_params = result.x[cam_start + n_cam_params:].reshape(n_points, 3)
        for i, pt_id in enumerate(point_indices):
            self.points3D[pt_id]['xyz'] = optimized_pt_params[i]
        
        print("  ✓ Bundle adjustment completed\n")
    
    # ============================================================================
    # SAVE & VISUALIZE
    # ============================================================================
    
    def save_results(self, output_dir):
        """Save results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'cameras.txt', 'w') as f:
            f.write("# Camera list\n")
            if self.cameras:
                K = self.cameras[0]['K']
                f.write(f"1 OPENCV 640 720 {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]} 0 0 0 0\n")
        
        with open(output_dir / 'images.txt', 'w') as f:
            f.write("# Image list\n")
            
            for img_idx in sorted(self.cameras.keys()):
                cam = self.cameras[img_idx]
                R = cam['R']
                t = cam['t'].ravel()
                
                rvec, _ = cv2.Rodrigues(R)
                angle = np.linalg.norm(rvec)
                if angle > 0:
                    axis = rvec.ravel() / angle
                    qw = np.cos(angle / 2)
                    qx, qy, qz = np.sin(angle / 2) * axis
                else:
                    qw, qx, qy, qz = 1, 0, 0, 0
                
                f.write(f"{img_idx+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 img_{img_idx:04d}.jpg\n\n")
        
        with open(output_dir / 'points3D.txt', 'w') as f:
            f.write("# 3D point list\n")
            
            for pt_id, pt_data in self.points3D.items():
                xyz = pt_data['xyz']
                rgb = pt_data['rgb']
                f.write(f"{pt_id} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} 0.0")
                
                for cam_idx, kp_idx, pt2d in pt_data['obs']:
                    f.write(f" {cam_idx+1} 0")
                f.write("\n")
        
        with open(output_dir / 'vis.txt', 'w') as f:
            for pt_data in self.points3D.values():
                xyz = pt_data['xyz']
                rgb = pt_data['rgb']
                f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
        
        print(f"Results saved to {output_dir}/")
    
    def visualize(self, output_dir):
        """Create visualization"""
        output_dir = Path(output_dir)
        
        vis_file = output_dir / 'vis.txt'
        if not vis_file.exists():
            return
        
        try:
            data = np.loadtxt(vis_file)
            if len(data) == 0:
                return
        except:
            return
        
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.savefig(output_dir / 'visualization.png', dpi=150)
        plt.close()
        
        print(f"Visualization saved")