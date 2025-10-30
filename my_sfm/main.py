from feature import FeatureDetector, ImageLoader, FeatureMatchGraph
from sfm_model import SfMModel
from feature import verify_feature_matching


def main():
    """Execute incremental SfM pipeline"""
    
    print("="*60)
    print("Incremental Structure from Motion - From Scratch")
    print("="*60 + "\n")
    
    # Configuration
    image_dir = 'data/frames'
    output_dir = 'results/my_sfm2'
    
    # Step 0: Feature Check (optional but recommended)
    print("STEP 0: Feature Matching Verification")
    print("-" * 60)
    run_verification = input("Run feature verification? (y/n): ").lower() == 'y'
    
    # Step 1: Feature Detection
    print("\nSTEP 1: Feature Detection and Matching")
    print("-" * 60)
    
    detector = FeatureDetector(ratio_thresh=0.75)
    loader = ImageLoader(image_dir)
    
    if len(loader) == 0:
        print("Error: No images found in", image_dir)
        return
    
    print(f"Loaded {len(loader)} images")
    
    # Extract features
    loader.extract_features(detector)
    
    if run_verification:
        verify_feature_matching(loader, detector)
    
    # Build match graph
    match_graph = FeatureMatchGraph(loader, detector)
    match_graph.build_match_graph()
    print(f"Built match graph with {len(match_graph.matches)} pairs\n")
    
    # Step 2: Initialize Reconstruction
    print("STEP 2: Initialize Reconstruction")
    print("-" * 60)
    
    # Select best initial pair
    initial_pair = match_graph.get_best_initial_pair()
    
    if initial_pair is None:
        print("Error: Could not find good initial pair")
        return
    
    idx1, idx2 = initial_pair
    
    # Get matched points
    matches = match_graph.get_matches(idx1, idx2)
    pts1, pts2 = detector.get_matched_points(
        loader.keypoints[idx1],
        loader.keypoints[idx2],
        matches
    )
    
    # Initialize model
    sfm = SfMModel()
    
    success = sfm.initialize_reconstruction(
        loader.load_image(idx1),
        loader.load_image(idx2),
        pts1, pts2, matches,
        loader.keypoints[idx1],
        loader.keypoints[idx2]
    )
    
    if not success:
        print("Failed to initialize reconstruction!")
        return
    
    # Step 3: Incremental Reconstruction
    print("STEP 3: Incremental Camera Registration")
    print("-" * 60)
    
    max_images = min(len(loader), 10)  # Limit for testing
    failed_images = set()  # Track failed images to avoid infinite loop
    max_failures = 3  # Allow max 3 consecutive failures
    consecutive_failures = 0
    
    while len(sfm.registered_images) < max_images:
        # Find next best image
        next_img, score = match_graph.get_next_image(sfm.registered_images)
        
        if next_img is None or score < 30:
            print("No more images can be registered\n")
            break
        
        # Skip if already failed multiple times
        if next_img in failed_images:
            print(f"Skipping image {next_img} (previously failed)\n")
            # Mark as "registered" to skip it in future searches
            sfm.registered_images.add(next_img)
            continue
        
        print(f"Next image: {next_img} (score: {score})")
        
        # Register image
        success = sfm.register_new_image(
            next_img,
            loader.load_image(next_img),
            loader.keypoints[next_img],
            match_graph,
            loader
        )
        
        if not success:
            print(f"Failed to register image {next_img}\n")
            failed_images.add(next_img)
            consecutive_failures += 1
            
            if consecutive_failures >= max_failures:
                print(f"Too many consecutive failures ({max_failures}). Stopping.\n")
                break
            continue
        
        # Reset failure counter on success
        consecutive_failures = 0
        
        # BA after initialization (structure only)
        if success:
            sfm.bundle_adjustment(optimize_intrinsics=False)

        # Incremental BA (every 3 images, with intrinsics)
        if len(sfm.registered_images) % 3 == 0:
            sfm.bundle_adjustment(optimize_intrinsics=True)

        # Final BA (full optimization)
        sfm.bundle_adjustment(optimize_intrinsics=True)
    
    # Remove failed images from registered set (they weren't actually registered)
    sfm.registered_images -= failed_images
    
    if len(sfm.registered_images) < 2:
        print("Error: Not enough images registered!")
        return

    
    # Step 5: Save Results
    print("STEP 5: Save Results")
    print("-" * 60)
    
    sfm.save_results(output_dir)
    print(f"\nFinal Statistics:")
    print(f"  Registered Images: {len(sfm.registered_images)}")
    print(f"  3D Points: {len(sfm.points3D)}")
    print(f"  Cameras: {len(sfm.cameras)}\n")
    
    # Step 6: Visualization
    print("STEP 6: Visualization")
    print("-" * 60)
    sfm.visualize(output_dir)
    
    print("\n" + "="*60)
    print("✨ SfM Complete! Check sfm_output/")
    print("="*60)


if __name__ == "__main__":
    main()