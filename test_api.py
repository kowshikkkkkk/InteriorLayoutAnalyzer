"""
Test script to demonstrate API usage.
Shows how to make requests to the API endpoints.

This can run while the server is running in another terminal.
"""

import requests
import json
from pathlib import Path


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint."""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure the server is running: python main.py")
        return False


def test_info():
    """Test info endpoint."""
    print("\n" + "="*70)
    print("TEST 2: Service Info")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/info")
        print(f"Status: {response.status_code}")
        info = response.json()
        print(f"Service: {info['service']}")
        print(f"Version: {info['version']}")
        print(f"Capabilities: {len(info['capabilities'])} features")
        for cap in info['capabilities']:
            print(f"  - {cap}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_analyze_image():
    """Test image analysis endpoint."""
    print("\n" + "="*70)
    print("TEST 3: Analyze Image")
    print("="*70)
    
    image_path = "data/raw/test_interior.jpg"
    
    if not Path(image_path).exists():
        print(f"✗ Test image not found: {image_path}")
        return False
    
    try:
        # Upload image
        print(f"Uploading image: {image_path}")
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        files = {'file': ('test_interior.jpg', image_data, 'image/jpeg')}
        response = requests.post(
            f"{BASE_URL}/analyze-image",
            files=files,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Analysis successful!")
            print(f"\nMetadata:")
            print(f"  Original size: {result['metadata']['original_size']}")
            print(f"  Processed size: {result['metadata']['processed_size']}")
            
            print(f"\nDetections:")
            print(f"  Total fixtures: {result['summary']['total_fixtures']}")
            print(f"  Fixture types: {result['summary']['fixture_types']}")
            print(f"  Room coverage: {result['summary']['room_coverage_percent']:.1f}%")
            
            print(f"\nDetected fixtures:")
            for det in result['detections']['fixtures'][:5]:
                print(f"  - {det['class_name']:15} (confidence: {det['confidence']:.1%})")
            
            print(f"\nClosest spatial relationships:")
            rels = result['spatial_analysis']['relationships'][:3]
            for rel in rels:
                print(f"  - {rel['object1']:15} {rel['relative_position']:12} {rel['object2']}")
            
            # Save full result
            with open("output/api_response.json", 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Full response saved to output/api_response.json")
            
            return True
        else:
            print(f"✗ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error - server not running")
        print("Start the server with: python main.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_analyze_with_visualization():
    """Test visualization endpoint."""
    print("\n" + "="*70)
    print("TEST 4: Analyze with Visualization")
    print("="*70)
    
    image_path = "data/raw/test_interior.jpg"
    
    if not Path(image_path).exists():
        print(f"✗ Test image not found: {image_path}")
        return False
    
    try:
        print(f"Uploading image: {image_path}")
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/analyze-with-visualization",
                files=files,
                timeout=60
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            fixtures_found = response.headers.get('X-Fixtures-Found', 'unknown')
            print(f"✓ Fixtures found: {fixtures_found}")
            
            # Save image
            output_path = "output/api_annotated.jpg"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Annotated image saved to {output_path}")
            
            return True
        else:
            print(f"✗ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error - server not running")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("INTERIOR LAYOUT ANALYZER - API TEST SUITE")
    print("="*70)
    print("\nNote: Make sure the server is running in another terminal:")
    print("  python main.py")
    
    print("\nTesting endpoints...\n")
    
    results = {
        "Health Check": test_health_check(),
        "Service Info": test_info(),
        "Analyze Image": test_analyze_image(),
        "Analyze with Visualization": test_analyze_with_visualization()
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ All tests passed!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()