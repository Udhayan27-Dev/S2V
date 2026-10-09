import cv2
import numpy as np

def check_quality(img_bgr, binary_img):
    """
    Check the quality of the image and its binary version.
    Returns a list of warnings.
    """
    warnings = []
    
    h, w = img_bgr.shape[:2]
    longer_side = max(h, w)
    
    # Resolution
    if longer_side < 1200:
        warnings.append("Low resolution: Image may lack detail (longer side is under 1200px).")
        
    # Blur check
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        warnings.append(f"Image appears blurry (Laplacian variance: {laplacian_var:.1f}).")
        
    # Coverage checks (Too faint / Too noisy)
    total_pixels = h * w
    stroke_pixels = total_pixels - cv2.countNonZero(binary_img)
    coverage = stroke_pixels / total_pixels
    
    if coverage < 0.001:
        warnings.append("Image is extremely faint: Very little stroke coverage detected.")
    elif coverage > 0.20:
        warnings.append("Image is extremely noisy: High stroke coverage detected.")
        
    return warnings
