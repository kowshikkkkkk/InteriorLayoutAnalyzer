"""
Test the YOLO object detection module.
This detects interior fixtures in the test image.
"""

from src.detection import FixtureDetector

print("=" * 60)
print("PHASE 3: YOLO OBJECT DETECTION TEST")
print("=" * 60)

try:
    # Initialize detector (downloads YOLOv8 nano model if not present)
    print("\nInitializing YOLOv8 detector...")
    print("(First run will download ~130MB model - this takes ~30-60 seconds)")
    detector = FixtureDetector(model_name='yolov8n.pt')
    
    # Run detection pipeline
    print("\nRunning detection pipeline on test image...")
    results = detector.detect_and_visualize('data/raw/test_interior.jpg')
    
    # Display results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"Total fixtures detected: {results['total_detected']}")
    
    # Get summary by class
    summary = detector.get_fixture_summary(results['detections'])
    print("\nFixtures by class:")
    for class_name, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {class_name}: {count}")
    
    # Show top 5 detections
    print("\nTop detections (by confidence):")
    top_detections = sorted(
        results['detections'],
        key=lambda x: x['confidence'],
        reverse=True
    )[:5]
    
    for i, det in enumerate(top_detections, 1):
        print(f"\n  {i}. {det['class_name']}")
        print(f"     Confidence: {det['confidence']:.2%}")
        print(f"     Position: {det['center']}")
        print(f"     Area: {det['area']:.0f} px²")
    
    print("\n" + "=" * 60)
    print("✓ Detection complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - output/detection_annotated.jpg (image with bounding boxes)")
    print("  - output/detections.json (structured results)")
    
except Exception as e:
    print(f"\n✗ Error during detection: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure test image exists: data/raw/test_interior.jpg")
    print("2. Check internet connection (model download may fail)")
    print("3. Try again - model download can be flaky on first attempt")