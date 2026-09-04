import math
import json
from pathlib import Path
from typing import List, Dict, Tuple


class SpatialAnalyzer:
    """
    Analyzes spatial relationships between detected fixtures.
    Calculates distances, positions, and relationships.
    """

    def __init__(self, image_width: int = 640, image_height: int = 640):
        """
        Initialize spatial analyzer.
        
        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
        """
        self.image_width = image_width
        self.image_height = image_height
        self.center_x = image_width / 2
        self.center_y = image_height / 2

    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: (x, y) coordinates
            point2: (x, y) coordinates
            
        Returns:
            Distance in pixels
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def determine_horizontal_position(self, center_x: float) -> str:
        """
        Determine if object is on LEFT, CENTER, or RIGHT of image.
        
        Args:
            center_x: X coordinate of object center
            
        Returns:
            Position string: 'left', 'center', or 'right'
        """
        third_width = self.image_width / 3
        
        if center_x < third_width:
            return 'left'
        elif center_x > 2 * third_width:
            return 'right'
        else:
            return 'center'

    def determine_vertical_position(self, center_y: float) -> str:
        """
        Determine if object is on TOP, MIDDLE, or BOTTOM of image.
        
        Args:
            center_y: Y coordinate of object center
            
        Returns:
            Position string: 'top', 'middle', or 'bottom'
        """
        third_height = self.image_height / 3
        
        if center_y < third_height:
            return 'top'
        elif center_y > 2 * third_height:
            return 'bottom'
        else:
            return 'middle'

    def get_absolute_position(self, center_x: float, center_y: float) -> str:
        """
        Get combined absolute position (e.g., 'top-left', 'center', 'bottom-right').
        
        Args:
            center_x: X coordinate
            center_y: Y coordinate
            
        Returns:
            Position string
        """
        h_pos = self.determine_horizontal_position(center_x)
        v_pos = self.determine_vertical_position(center_y)
        
        if h_pos == 'center' and v_pos == 'middle':
            return 'center'
        else:
            return f"{v_pos}-{h_pos}"

    def determine_relative_position(self, point1: Tuple[float, float], 
                                   point2: Tuple[float, float]) -> str:
        """
        Determine relative position of point2 with respect to point1.
        
        Args:
            point1: (x, y) reference point
            point2: (x, y) other point
            
        Returns:
            Relationship: 'above', 'below', 'left_of', 'right_of', 'diagonal'
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Determine primary direction based on larger difference
        if abs_dy > abs_dx * 1.5:  # Primarily vertical
            return 'above' if dy < 0 else 'below'
        elif abs_dx > abs_dy * 1.5:  # Primarily horizontal
            return 'left_of' if dx < 0 else 'right_of'
        else:  # Diagonal
            diagonal_dir = f"{'above' if dy < 0 else 'below'}-{'left' if dx < 0 else 'right'}"
            return diagonal_dir

    def calculate_proximity(self, distance: float, max_distance: float = 300) -> str:
        """
        Categorize distance as 'very_close', 'close', 'medium', 'far'.
        
        Args:
            distance: Distance in pixels
            max_distance: Threshold for 'far' distance
            
        Returns:
            Proximity category
        """
        if distance < 50:
            return 'touching'
        elif distance < 100:
            return 'very_close'
        elif distance < 200:
            return 'close'
        elif distance < max_distance:
            return 'medium'
        else:
            return 'far'

    def analyze_pairwise_relationships(self, detections: List[Dict]) -> List[Dict]:
        """
        Calculate spatial relationships between all pairs of objects.
        
        Args:
            detections: List of detection objects
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        for i, det1 in enumerate(detections):
            for det2 in detections[i+1:]:
                center1 = tuple(det1['center'])
                center2 = tuple(det2['center'])
                
                distance = self.calculate_distance(center1, center2)
                relative_pos = self.determine_relative_position(center1, center2)
                proximity = self.calculate_proximity(distance)
                
                relationship = {
                    'object1': det1['class_name'],
                    'object2': det2['class_name'],
                    'distance': round(distance, 2),
                    'relative_position': relative_pos,
                    'proximity': proximity,
                    'confidence1': round(det1['confidence'], 3),
                    'confidence2': round(det2['confidence'], 3)
                }
                
                relationships.append(relationship)
        
        # Sort by distance (closest first)
        relationships.sort(key=lambda x: x['distance'])
        
        return relationships

    def analyze_layout_distribution(self, detections: List[Dict]) -> Dict:
        """
        Analyze overall distribution of objects in the room.
        
        Args:
            detections: List of detections
            
        Returns:
            Distribution analysis
        """
        left_objects = []
        center_objects = []
        right_objects = []
        top_objects = []
        middle_objects = []
        bottom_objects = []
        
        for det in detections:
            center_x, center_y = det['center']
            class_name = det['class_name']
            
            # Horizontal distribution
            h_pos = self.determine_horizontal_position(center_x)
            if h_pos == 'left':
                left_objects.append(class_name)
            elif h_pos == 'center':
                center_objects.append(class_name)
            else:
                right_objects.append(class_name)
            
            # Vertical distribution
            v_pos = self.determine_vertical_position(center_y)
            if v_pos == 'top':
                top_objects.append(class_name)
            elif v_pos == 'middle':
                middle_objects.append(class_name)
            else:
                bottom_objects.append(class_name)
        
        distribution = {
            'horizontal': {
                'left': left_objects,
                'center': center_objects,
                'right': right_objects
            },
            'vertical': {
                'top': top_objects,
                'middle': middle_objects,
                'bottom': bottom_objects
            },
            'total_objects': len(detections),
            'unique_types': len(set(d['class_name'] for d in detections))
        }
        
        return distribution

    def calculate_room_coverage(self, detections: List[Dict]) -> Dict:
        """
        Calculate how much of the room is occupied by detected objects.
        
        Args:
            detections: List of detections
            
        Returns:
            Coverage statistics
        """
        total_area = self.image_width * self.image_height
        objects_area = sum(det['area'] for det in detections)
        coverage_percent = (objects_area / total_area) * 100
        
        # Calculate average object size
        avg_area = objects_area / len(detections) if detections else 0
        
        # Categorize objects by size
        large_objects = [d for d in detections if d['area'] > avg_area * 1.5]
        medium_objects = [d for d in detections if avg_area * 0.7 <= d['area'] <= avg_area * 1.5]
        small_objects = [d for d in detections if d['area'] < avg_area * 0.7]
        
        coverage = {
            'total_objects_area': round(objects_area, 2),
            'room_area': total_area,
            'coverage_percent': round(coverage_percent, 2),
            'average_object_size': round(avg_area, 2),
            'large_objects': len(large_objects),
            'medium_objects': len(medium_objects),
            'small_objects': len(small_objects)
        }
        
        return coverage

    def generate_layout_summary(self, detections: List[Dict]) -> Dict:
        """
        Generate complete spatial analysis summary.
        
        Args:
            detections: List of detections
            
        Returns:
            Complete analysis with all metrics
        """
        summary = {
            'total_fixtures_detected': len(detections),
            'fixtures': [],
            'relationships': self.analyze_pairwise_relationships(detections),
            'distribution': self.analyze_layout_distribution(detections),
            'coverage': self.calculate_room_coverage(detections)
        }
        
        # Add position analysis for each fixture
        for det in detections:
            fixture_info = {
                'class': det['class_name'],
                'confidence': round(det['confidence'], 3),
                'center': [round(c, 2) for c in det['center']],
                'area': round(det['area'], 2),
                'dimensions': {
                    'width': round(det['width'], 2),
                    'height': round(det['height'], 2)
                },
                'position': self.get_absolute_position(det['center'][0], det['center'][1])
            }
            summary['fixtures'].append(fixture_info)
        
        return summary

    def save_analysis_json(self, analysis: Dict, output_path: str):
        """
        Save spatial analysis to JSON file.
        
        Args:
            analysis: Analysis dictionary
            output_path: Path to save JSON
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Analysis saved to {output_path}")

    def print_analysis_summary(self, analysis: Dict):
        """
        Print human-readable summary of analysis.
        
        Args:
            analysis: Analysis dictionary
        """
        print("\n" + "="*70)
        print("SPATIAL ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nTotal fixtures detected: {analysis['total_fixtures_detected']}")
        
        # Fixture positions
        print("\nFixture Positions:")
        for fixture in analysis['fixtures']:
            print(f"  • {fixture['class']:15} - {fixture['position']:15} " +
                  f"(confidence: {fixture['confidence']:.1%})")
        
        # Close relationships
        print("\nClosest relationships:")
        for rel in analysis['relationships'][:5]:
            print(f"  • {rel['object1']:15} {rel['relative_position']:12} " +
                  f"{rel['object2']:15} ({rel['proximity']}, {rel['distance']}px)")
        
        # Distribution
        dist = analysis['distribution']
        print(f"\nHorizontal distribution: " +
              f"Left({len(dist['horizontal']['left'])}) | " +
              f"Center({len(dist['horizontal']['center'])}) | " +
              f"Right({len(dist['horizontal']['right'])})")
        
        print(f"Vertical distribution: " +
              f"Top({len(dist['vertical']['top'])}) | " +
              f"Middle({len(dist['vertical']['middle'])}) | " +
              f"Bottom({len(dist['vertical']['bottom'])})")
        
        # Coverage
        cov = analysis['coverage']
        print(f"\nRoom coverage: {cov['coverage_percent']:.1f}%")
        print(f"Object size distribution: " +
              f"Large({cov['large_objects']}) | " +
              f"Medium({cov['medium_objects']}) | " +
              f"Small({cov['small_objects']})")
        
        print("\n" + "="*70)