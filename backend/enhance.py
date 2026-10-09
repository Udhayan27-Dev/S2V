import cv2
import numpy as np

def enhance(img_bgr, black_pct=82.0, gamma=0.9, stroke_px=None):
    """
    Enhance a hand-drawn pencil sketch.
    Returns: (enhanced_gray, binary)
    """
    # 1. Convert to grayscale, Gaussian blur sigma ~1.2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    
    # 2. Exposure/highlights: estimate paper brightness
    h, w = blurred.shape
    longer_side = max(h, w)
    
    # Morphological closing (kernel ~6% of longer side)
    close_kernel_size = int(longer_side * 0.06)
    if close_kernel_size % 2 == 0:
        close_kernel_size += 1
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    paper_estimate = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_close)
    paper_estimate = cv2.GaussianBlur(paper_estimate, (0, 0), sigmaX=close_kernel_size/3.0, sigmaY=close_kernel_size/3.0)
    
    # flat-field: divide image by it and clip 0..1
    paper_estimate = paper_estimate.astype(np.float32)
    paper_estimate[paper_estimate == 0] = 1.0
    flat_field = (blurred.astype(np.float32) / paper_estimate)
    flat_field = np.clip(flat_field, 0, 1.0)
    
    # Convert back to 0-255 for next morphology steps
    flat_field_8u = (flat_field * 255).astype(np.uint8)
    
    # 3. Keep only thin dark strokes: black-hat morphology
    if stroke_px is None:
        stroke_px = int(longer_side * 0.012)
        if stroke_px < 15:
            stroke_px = 15
    if stroke_px % 2 == 0:
        stroke_px += 1
        
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_px, stroke_px))
    blackhat = cv2.morphologyEx(flat_field_8u, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # 4. Black point / white point: ignore weakest ~90% of stroke signal
    black_pt = np.percentile(blackhat, black_pct)
    white_pt = np.percentile(blackhat, 99.5)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        normalized = (blackhat.astype(np.float32) - black_pt) / (white_pt - black_pt)
    normalized = np.clip(normalized, 0, 1.0)
    
    # 5. Gamma (default 0.7) to lift faint pencil lines
    gamma_corrected = np.power(normalized, gamma)
    
    # 6. Invert so strokes are dark on white paper -> enhanced_gray
    enhanced_gray = 1.0 - gamma_corrected
    enhanced_gray_8u = (enhanced_gray * 255).astype(np.uint8)
    
    # 7. Otsu threshold
    _, binary = cv2.threshold(enhanced_gray_8u, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # remove small connected components (specks)
    inv_binary = cv2.bitwise_not(binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_binary, connectivity=8)
    
    min_area = max(4, int((longer_side * 0.002) ** 2))
    
    out_inv_binary = np.zeros_like(inv_binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out_inv_binary[labels == i] = 255
            
    final_binary = cv2.bitwise_not(out_inv_binary)
    
    return enhanced_gray_8u, final_binary
