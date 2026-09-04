"""
Quick test of preprocessing module.
This verifies all functions work correctly.
"""

import cv2
from src.preprocessing import ImagePreprocessor

# Test 1: Load a sample image
print("=" * 50)
print("TEST 1: Load Sample Image")
print("=" * 50)

# First, we need a test image. Let's create a dummy one.
print("Creating a test interior image (640x480)...")

# Create a simple test image (simulating an interior photo)
test_image = cv2.imread('data/raw/test_interior.jpg')

if test_image is None:
    print("⚠️  No test image found at data/raw/test_interior.jpg")
    print("Please add an interior photo to data/raw/ folder")
    print("\nYou can download a free indoor image from:")
    print("- https://unsplash.com/ (search: interior, living room, kitchen)")
    print("- https://www.pexels.com/ (search: indoor furniture)")
    print("\nOnce you add an image, run this test again.")
else:
    print(f"✓ Loaded test image: shape {test_image.shape}")
    
    preprocessor = ImagePreprocessor()
    
    # Test 2: Resize
    print("\n" + "=" * 50)
    print("TEST 2: Resize Image")
    print("=" * 50)
    resized = preprocessor.resize_image(test_image, 640, 640)
    print(f"✓ Resized to: {resized.shape}")
    
    # Test 3: Convert to grayscale
    print("\n" + "=" * 50)
    print("TEST 3: Convert to Grayscale")
    print("=" * 50)
    gray = preprocessor.to_grayscale(resized)
    print(f"✓ Grayscale shape: {gray.shape}")
    
    # Test 4: Gaussian blur
    print("\n" + "=" * 50)
    print("TEST 4: Apply Gaussian Blur")
    print("=" * 50)
    blurred = preprocessor.gaussian_blur(gray, kernel_size=5)
    print(f"✓ Blurred shape: {blurred.shape}")
    
    # Test 5: Edge detection
    print("\n" + "=" * 50)
    print("TEST 5: Canny Edge Detection")
    print("=" * 50)
    edges = preprocessor.canny_edge_detection(blurred)
    print(f"✓ Edges shape: {edges.shape}")
    
    # Test 6: Find contours
    print("\n" + "=" * 50)
    print("TEST 6: Find Contours")
    print("=" * 50)
    contours, hierarchy = preprocessor.find_contours(edges)
    print(f"✓ Found {len(contours)} contours")
    
    # Test 7: Save results
    print("\n" + "=" * 50)
    print("TEST 7: Save Preprocessed Images")
    print("=" * 50)
    preprocessor.save_image(resized, "output/test_resized.jpg")
    preprocessor.save_image(gray, "output/test_grayscale.jpg")
    preprocessor.save_image(blurred, "output/test_blurred.jpg")
    preprocessor.save_image(edges, "output/test_edges.jpg")
    
    # Test 8: Full pipeline
    print("\n" + "=" * 50)
    print("TEST 8: Full Preprocessing Pipeline")
    print("=" * 50)
    try:
        results = preprocessor.preprocess_pipeline('data/raw/test_interior.jpg')
        print(f"✓ Pipeline complete! {len(results)} outputs generated")
        print(f"  - Original size: {results['original_size']}")
        print(f"  - Contours found: {len(results['contours'])}")
    except Exception as e:
        print(f"✗ Pipeline error: {e}")

print("\n" + "=" * 50)
print("✓ All tests complete!")
print("=" * 50)