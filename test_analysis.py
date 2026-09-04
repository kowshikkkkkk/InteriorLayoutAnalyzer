"""
Test spatial analysis on detection results.
"""

import json
from src.analysis import SpatialAnalyzer

print("=" * 70)
print("PHASE 4: SPATIAL ANALYSIS TEST")
print("=" * 70)

# Load detections from previous phase
try:
    with open('output/detections.json', 'r') as f:
        detections = json.load(f)
    
    print(f"\nLoaded {len(detections)} detections from output/detections.json")
    
    # Get image dimensions from original detections
    # Use the actual dimensions from the first detection
    if detections:
        max_x = max(d['bbox']['x2'] for d in detections)
        max_y = max(d['bbox']['y2'] for d in detections)
        print(f"Image dimensions: {max_x:.0f} x {max_y:.0f}")
        image_width = int(max_x)
        image_height = int(max_y)
    else:
        image_width = 640
        image_height = 640
    
    # Initialize analyzer
    analyzer = SpatialAnalyzer(image_width, image_height)
    
    # Generate analysis
    print("\nAnalyzing spatial relationships...")
    analysis = analyzer.generate_layout_summary(detections)
    
    # Print summary
    analyzer.print_analysis_summary(analysis)
    
    # Save results
    analyzer.save_analysis_json(analysis, 'output/spatial_analysis.json')
    
    print("\n✓ Spatial analysis complete!")
    print("\nGenerated file:")
    print("  - output/spatial_analysis.json (spatial relationships & distribution)")
    
except FileNotFoundError:
    print("\n✗ Error: output/detections.json not found")
    print("Please run test_detection.py first to generate detections")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()