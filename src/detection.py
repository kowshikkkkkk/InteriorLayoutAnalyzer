import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json


class FixtureDetector:
    """
    Detects interior fixtures using YOLOv8.
    Identifies furniture, doors, windows, appliances, etc.
    """

    # Classes we care about for interior analysis
    INTERIOR_CLASSES = {
        'person': 0,           # People in the room
        'chair': 1,            # Chairs
        'table': 2,            # Tables
        'sofa': 3,             # Sofas/couches
        'bed': 4,              # Beds
        'potted plant': 5,     # Plants
        'dining table': 6,     # Dining tables
        'bench': 7,            # Benches
        'backpack': 8,         # Bags
        'umbrella': 9,         # Umbrellas
        'handbag': 10,         # Handbags
        'suitcase': 11,        # Suitcases
        'microwave': 12,       # Microwave
        'oven': 13,            # Oven
        'sink': 14,            # Sink (plumbing fixture)
        'refrigerator': 15,    # Fridge
        'book': 16,            # Books
        'clock': 17,           # Clocks
        'cup': 18,             # Cups/mugs
        'bowl': 19,            # Bowls
        'bottle': 20,          # Bottles
        'tvmonitor': 21,       # TV/Monitor
        'laptop': 22,          # Laptops
        'mouse': 23,           # Mouse
        'keyboard': 24,        # Keyboard
        'vase': 25,            # Vases
        'scissors': 26,        # Scissors
        'teddy bear': 27,      # Teddy bears
        'hair drier': 28,      # Hair dryer
        'toothbrush': 29,      # Toothbrush
        'door': 30,            # Doors
        'window': 31,          # Windows
        'wall': 32,            # Walls
        'floor': 33,           # Floors
        'cabinet': 34,         # Cabinets
        'counter': 35,         # Countertops
    }

    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLOv8 model variant
                - 'yolov8n.pt' (nano - fastest, smallest)
                - 'yolov8s.pt' (small)
                - 'yolov8m.pt' (medium)
                - 'yolov8l.pt' (large - most accurate)
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        print(f"✓ Model loaded successfully")

    def detect(self, image, confidence_threshold=0.5):
        """
        Detect objects in image using YOLOv8.
        
        Args:
            image: Input image (numpy array or file path)
            confidence_threshold: Only keep detections above this confidence (0-1)
            
        Returns:
            List of detections with class, confidence, and bbox
        """
        # Run inference
        results = self.model(image, conf=confidence_threshold, verbose=False)
        
        # Extract detections from results
        detections = []
        
        for result in results:
            # result.boxes contains all detected boxes for this image
            for box in result.boxes:
                detection = {
                    'class_id': int(box.cls),
                    'class_name': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': {
                        'x1': float(box.xyxy[0][0]),
                        'y1': float(box.xyxy[0][1]),
                        'x2': float(box.xyxy[0][2]),
                        'y2': float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
        
        return detections

    def filter_interior_fixtures(self, detections, include_classes=None):
        """
        Filter detections to only interior-relevant classes.
        
        Args:
            detections: List of detections from detect()
            include_classes: Custom list of classes to include (if None, uses default)
            
        Returns:
            Filtered detections
        """
        if include_classes is None:
            include_classes = [
                'chair', 'table', 'sofa', 'bed', 'potted plant',
                'dining table', 'bench', 'microwave', 'oven', 'sink',
                'refrigerator', 'tvmonitor', 'cabinet', 'counter',
                'door', 'window', 'vase', 'person'
            ]
        
        filtered = [
            d for d in detections 
            if d['class_name'].lower() in [c.lower() for c in include_classes]
        ]
        
        return filtered

    def calculate_bbox_center(self, bbox):
        """
        Calculate center point of bounding box.
        
        Args:
            bbox: Bounding box dict with x1, y1, x2, y2
            
        Returns:
            (center_x, center_y) tuple
        """
        center_x = (bbox['x1'] + bbox['x2']) / 2
        center_y = (bbox['y1'] + bbox['y2']) / 2
        return (center_x, center_y)

    def calculate_bbox_area(self, bbox):
        """
        Calculate area of bounding box.
        
        Args:
            bbox: Bounding box dict
            
        Returns:
            Area in pixels squared
        """
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        area = width * height
        return area

    def calculate_bbox_dimensions(self, bbox):
        """
        Calculate width and height of bounding box.
        
        Args:
            bbox: Bounding box dict
            
        Returns:
            (width, height) tuple
        """
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        return (width, height)

    def enrich_detections(self, detections):
        """
        Add calculated fields to detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Detections with added center, area, dimensions
        """
        for det in detections:
            det['center'] = self.calculate_bbox_center(det['bbox'])
            det['area'] = self.calculate_bbox_area(det['bbox'])
            det['width'], det['height'] = self.calculate_bbox_dimensions(det['bbox'])
        
        return detections

    def draw_detections(self, image, detections, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: List of detections
            color: BGR color for boxes
            thickness: Line thickness
            
        Returns:
            Image with drawn boxes
        """
        output = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Extract coordinates
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return output

    def save_detections_json(self, detections, output_path):
        """
        Save detections to JSON file.
        
        Args:
            detections: List of detections
            output_path: Path to save JSON
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        
        print(f"✓ Detections saved to {output_path}")

    def detect_and_visualize(self, image_path, output_dir='output/'):
        """
        Complete detection pipeline:
        Load → Detect → Filter → Enrich → Draw → Save
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            Dictionary with detections and visualization
        """
        # Load image
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Detect all objects
        print("Detecting objects...")
        all_detections = self.detect(image, confidence_threshold=0.4)
        print(f"  Found {len(all_detections)} total objects")
        
        # Filter to interior fixtures
        print("Filtering to interior fixtures...")
        interior_detections = self.filter_interior_fixtures(all_detections)
        print(f"  Found {len(interior_detections)} interior fixtures")
        
        # Enrich with calculated fields
        interior_detections = self.enrich_detections(interior_detections)
        
        # Draw on image
        annotated = self.draw_detections(image, interior_detections)
        
        # Save results
        output_image_path = f"{output_dir}/detection_annotated.jpg"
        output_json_path = f"{output_dir}/detections.json"
        
        cv2.imwrite(output_image_path, annotated)
        self.save_detections_json(interior_detections, output_json_path)
        
        print(f"✓ Annotated image saved to {output_image_path}")
        
        results = {
            'image': image,
            'annotated': annotated,
            'detections': interior_detections,
            'total_detected': len(interior_detections)
        }
        
        return results

    def get_fixture_summary(self, detections):
        """
        Summarize detected fixtures by class.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with counts per class
        """
        summary = {}
        for det in detections:
            class_name = det['class_name']
            summary[class_name] = summary.get(class_name, 0) + 1
        
        return summary