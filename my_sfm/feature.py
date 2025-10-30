import cv2
import numpy as np
from pathlib import Path


class FeatureDetector:
    """
    SIFT feature detection and matching
    Based on lecture: SIFT + ratio test (0.8) + RANSAC refinement
    """
    
    def __init__(self, ratio_thresh=0.75):
        self.sift = cv2.SIFT_create()
        self.ratio_thresh = ratio_thresh  # 0.75 is more conservative than Lowe's 0.8
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Match features using ratio test
        Based on Lowe's paper: ratio_thresh=0.8
        """
        if desc1 is None or desc2 is None:
            return []
        
        # KNN match with k=2
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def get_matched_points(self, kp1, kp2, matches):
        """Extract matched point coordinates"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2


class ImageLoader:
    """Load and manage image sequences"""
    
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(self.image_dir.glob('*.jpg')) + \
                          sorted(self.image_dir.glob('*.png'))
        self.images = {}
        self.keypoints = {}
        self.descriptors = {}
    
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, idx):
        """Load image by index"""
        if idx not in self.images:
            img_path = self.image_paths[idx]
            self.images[idx] = cv2.imread(str(img_path))
        return self.images[idx]
    
    def get_image_name(self, idx):
        """Get image filename"""
        return self.image_paths[idx].name
    
    def extract_features(self, detector):
        """Extract features for all images"""
        for idx in range(len(self)):
            img = self.load_image(idx)
            kp, desc = detector.detect_and_compute(img)
            self.keypoints[idx] = kp
            self.descriptors[idx] = desc


class FeatureMatchGraph:
    """
    Build feature match graph between images
    Following lecture: RANSAC-refined matching for each pair
    """
    
    def __init__(self, image_loader, detector):
        self.loader = image_loader
        self.detector = detector
        self.matches = {}  # (i, j) -> list of DMatch
        self.match_counts = {}  # (i, j) -> int
    
    def build_match_graph(self):
        """Build match graph with RANSAC-refined matches"""
        n_images = len(self.loader)
        
        for i in range(n_images):
            for j in range(i + 1, n_images):
                # Get initial matches
                matches = self.detector.match_features(
                    self.loader.descriptors[i],
                    self.loader.descriptors[j]
                )
                
                if len(matches) < 20:
                    continue
                
                # Refine with RANSAC (fundamental matrix)
                pts1 = np.float32([self.loader.keypoints[i][m.queryIdx].pt for m in matches])
                pts2 = np.float32([self.loader.keypoints[j][m.trainIdx].pt for m in matches])
                
                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
                
                if mask is not None:
                    inliers = mask.ravel().astype(bool)
                    good_matches = [m for k, m in enumerate(matches) if inliers[k]]
                    
                    if len(good_matches) > 20:
                        self.matches[(i, j)] = good_matches
                        self.match_counts[(i, j)] = len(good_matches)
    
    def get_best_initial_pair(self):
        """Select best initial image pair"""
        if not self.match_counts:
            return None
        
        best_pair = max(self.match_counts.items(), key=lambda x: x[1])
        return best_pair[0]
    
    def get_matches(self, i, j):
        """Get matches between image i and j"""
        if i > j:
            i, j = j, i
        return self.matches.get((i, j), [])
    
    def get_next_image(self, registered_images):
        """Find next best image to register"""
        best_img = None
        best_score = 0
        
        for idx in range(len(self.loader)):
            if idx in registered_images:
                continue
            
            # Count total matches with registered images
            total_matches = 0
            for reg_idx in registered_images:
                i, j = min(idx, reg_idx), max(idx, reg_idx)
                total_matches += self.match_counts.get((i, j), 0)
            
            if total_matches > best_score:
                best_score = total_matches
                best_img = idx
        
        return best_img, best_score
    


def visualize_matches(img1, img2, kp1, kp2, matches, save_path, max_display=50):
    """Draw feature matches between two images"""
    # Sample matches if too many
    if len(matches) > max_display:
        matches_display = np.random.choice(matches, max_display, replace=False).tolist()
    else:
        matches_display = matches
    
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_display, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.imwrite(save_path, img_matches)
    print(f"  Saved: {save_path}")


def verify_feature_matching(loader, detector, output_dir='feature_check'):
    """
    Verify feature detection and matching quality
    Tests the best pair to diagnose issues
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("FEATURE MATCHING VERIFICATION")
    print("="*60)
    
    # Test first pair
    idx1, idx2 = 0, 1
    
    img1 = loader.load_image(idx1)
    img2 = loader.load_image(idx2)
    kp1 = loader.keypoints[idx1]
    kp2 = loader.keypoints[idx2]
    desc1 = loader.descriptors[idx1]
    desc2 = loader.descriptors[idx2]
    
    print(f"\nTesting pair [{idx1}] - [{idx2}]:")
    print(f"  Image 1: {img1.shape}, {len(kp1)} features")
    print(f"  Image 2: {img2.shape}, {len(kp2)} features")
    
    # Step 1: Raw matches (before ratio test)
    raw_matches = detector.bf_matcher.knnMatch(desc1, desc2, k=2)
    print(f"  Raw matches (k=2): {len(raw_matches)}")
    
    # Step 2: After ratio test
    good_matches = detector.match_features(desc1, desc2)
    print(f"  After ratio test (0.80): {len(good_matches)} matches")
    
    # Step 3: RANSAC with Fundamental matrix
    if len(good_matches) >= 8:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        if mask is not None:
            inliers = mask.ravel().astype(bool)
            print(f"  After RANSAC: {inliers.sum()} inliers / {len(good_matches)}")
            print(f"  Inlier ratio: {inliers.sum()/len(good_matches)*100:.1f}%")
            
            # Visualize
            visualize_matches(
                img1, img2, kp1, kp2, 
                [m for i, m in enumerate(good_matches) if inliers[i]],
                str(output_dir / f'matches_{idx1}_{idx2}_inliers.jpg')
            )
        else:
            print("  RANSAC failed!")
    else:
        print(f"  Too few matches for RANSAC!")
    
    # Save raw matches for comparison
    visualize_matches(
        img1, img2, kp1, kp2, good_matches,
        str(output_dir / f'matches_{idx1}_{idx2}_raw.jpg')
    )
    
    print("\n" + "="*60 + "\n")