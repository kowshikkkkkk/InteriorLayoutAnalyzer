import cv2
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """
    Handles all image preprocessing operations.
    Converts raw images into normalized format for detection.
    """

    @staticmethod
    def load_image(image_path):
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array (BGR format from OpenCV)
            
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image cannot be read
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        return image

    @staticmethod
    def resize_image(image, width=640, height=640):
        """
        Resize image to specified dimensions.
        
        Args:
            image: Input image (numpy array)
            width: Target width in pixels
            height: Target height in pixels
            
        Returns:
            Resized image
        """
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def bgr_to_rgb(image):
        """
        Convert BGR (OpenCV default) to RGB.
        
        Args:
            image: Input image in BGR
            
        Returns:
            Image in RGB format
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb

    @staticmethod
    def to_grayscale(image):
        """
        Convert image to grayscale.
        Useful for edge detection and contour finding.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Grayscale image (single channel)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def to_hsv(image):
        """
        Convert image to HSV color space.
        HSV is better for detecting specific colors (useful for furniture detection).
        
        Args:
            image: Input image (BGR)
            
        Returns:
            HSV image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv

    @staticmethod
    def gaussian_blur(image, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            image: Input image
            kernel_size: Size of blur kernel (must be odd, e.g., 3, 5, 7)
            sigma: Standard deviation
            
        Returns:
            Blurred image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred

    @staticmethod
    def median_blur(image, kernel_size=5):
        """
        Apply median blur (better for salt-and-pepper noise).
        
        Args:
            image: Input image
            kernel_size: Size of blur kernel (must be odd)
            
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.medianBlur(image, kernel_size)
        return blurred

    @staticmethod
    def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
        """
        Adjust brightness and contrast of image.
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
            
        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted

    @staticmethod
    def normalize_image(image):
        """
        Normalize image to 0-1 range (useful for deep learning models).
        
        Args:
            image: Input image (0-255)
            
        Returns:
            Normalized image (0-1)
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized

    @staticmethod
    def canny_edge_detection(image, threshold1=100, threshold2=200):
        """
        Detect edges using Canny edge detector.
        Useful for finding walls, doors, furniture outlines.
        
        Args:
            image: Input image (preferably grayscale)
            threshold1: Lower threshold
            threshold2: Upper threshold
            
        Returns:
            Binary edge map
        """
        # If color image, convert to grayscale first
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(image, threshold1, threshold2)
        return edges

    @staticmethod
    def find_contours(image):
        """
        Find contours in binary image.
        Useful for identifying object boundaries.
        
        Args:
            image: Binary image (result of edge detection or thresholding)
            
        Returns:
            List of contours and hierarchy
        """
        # If color image, convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    @staticmethod
    def draw_contours(image, contours, color=(0, 255, 0), thickness=2):
        """
        Draw contours on image.
        
        Args:
            image: Input image
            contours: List of contours
            color: BGR color tuple (default: green)
            thickness: Line thickness
            
        Returns:
            Image with drawn contours
        """
        output = image.copy()
        cv2.drawContours(output, contours, -1, color, thickness)
        return output

    @staticmethod
    def save_image(image, output_path):
        """
        Save image to file.
        
        Args:
            image: Image to save
            output_path: Path where to save the image
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"✓ Image saved to {output_path}")

    @staticmethod
    def preprocess_pipeline(image_path, output_dir="output/"):
        """
        Complete preprocessing pipeline.
        Load → Resize → Blur → Edge detect → Save intermediate results
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save intermediate results
            
        Returns:
            Dictionary with preprocessing results
        """
        preprocessor = ImagePreprocessor()
        
        # Load
        print(f"Loading image: {image_path}")
        image = preprocessor.load_image(image_path)
        original_h, original_w = image.shape[:2]
        
        # Resize
        resized = preprocessor.resize_image(image, 640, 640)
        
        # Blur
        blurred = preprocessor.gaussian_blur(resized, kernel_size=5)
        
        # Grayscale
        gray = preprocessor.to_grayscale(blurred)
        
        # Edge detection
        edges = preprocessor.canny_edge_detection(gray)
        
        # Contours
        contours, _ = preprocessor.find_contours(edges)
        contour_image = preprocessor.draw_contours(resized, contours)
        
        # Save intermediate results
        preprocessor.save_image(resized, f"{output_dir}/01_resized.jpg")
        preprocessor.save_image(blurred, f"{output_dir}/02_blurred.jpg")
        preprocessor.save_image(gray, f"{output_dir}/03_grayscale.jpg")
        preprocessor.save_image(edges, f"{output_dir}/04_edges.jpg")
        preprocessor.save_image(contour_image, f"{output_dir}/05_contours.jpg")
        
        results = {
            "original": image,
            "original_size": (original_w, original_h),
            "resized": resized,
            "blurred": blurred,
            "grayscale": gray,
            "edges": edges,
            "contours": contours,
            "contour_image": contour_image
        }
        
        return results