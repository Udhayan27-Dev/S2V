import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from page import detect_page, straighten
from enhance import enhance
from geometry import extract_geometry

def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = '/home/udhayan/.gemini/antigravity/brain/d1b1b3d8-4505-4a9c-9123-8d2e28cf7181/media__1791487731375.png'
    print(f"Loading {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Failed to load image")
        return

    # Phase 2: Straighten
    corners, conf, method, warnings = detect_page(img_bgr)
    straightened = straighten(img_bgr, corners)
    
    # Phase 3: Enhance
    bin_img, enhanced_gray = enhance(straightened)
    
    # Phase 4: Geometry
    geometry = extract_geometry(bin_img, enhanced_gray, {})
    
    segments = geometry.get("segments", [])
    unknown_roles = [s for s in segments if s.get("role") == "unknown"]
    
    print(f"--- RESULTS ---")
    print(f"Number of segments after Geometry extraction: {len(segments)}")
    print(f"Segments with role 'unknown': {len(unknown_roles)}")
    
    # Find collinear fragments roughly
    collinear = 0
    # A fragment is likely collinear if it has the same orientation and role outline, and its endpoints are close to the line of another
    
    
if __name__ == '__main__':
    main()
