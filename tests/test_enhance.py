import sys
import os
import cv2
import numpy as np

# Add backend to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from enhance import enhance
from quality import check_quality

def test_enhance_shapes_and_types():
    # Create a synthetic image
    img = np.ones((1500, 2000, 3), dtype=np.uint8) * 200
    cv2.line(img, (100, 100), (1900, 1400), (50, 50, 50), 3)
    
    enhanced, binary = enhance(img, black_pct=90.0, gamma=0.7, stroke_px=None)
    
    assert enhanced.shape == (1500, 2000)
    assert enhanced.dtype == np.uint8
    assert binary.shape == (1500, 2000)
    assert binary.dtype == np.uint8
    
    warnings = check_quality(img, binary)
    assert isinstance(warnings, list)

if __name__ == "__main__":
    test_enhance_shapes_and_types()
    print("Tests passed.")
