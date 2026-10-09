import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
import page

def create_synthetic_page():
    # Create dark background (orange table)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    img[:] = (50, 150, 255)
    
    # Paper corners (skewed)
    corners = np.array([
        [200, 150],  # TL
        [850, 200],  # TR
        [900, 850],  # BR
        [150, 900]   # BL
    ], dtype=np.int32)
    
    # Draw white paper
    cv2.fillPoly(img, [corners], (240, 250, 255))
    
    # Draw some random lines inside to simulate sketch
    cv2.line(img, (300, 300), (800, 800), (50, 50, 50), 3)
    cv2.line(img, (700, 300), (400, 800), (50, 50, 50), 3)
    
    return img, corners

def test_detect_page():
    img, gt_corners = create_synthetic_page()
    corners, conf, method, warnings = page.detect_page(img)
    
    assert conf > 0.5
    assert len(corners) == 4
    
    det = page.order_corners(np.array(corners))
    gt = page.order_corners(np.array(gt_corners))
    
    diff = np.linalg.norm(det - gt, axis=1)
    assert np.all(diff < 50)

def test_order_corners():
    pts = np.array([
        [900, 900], # BR
        [100, 100], # TL
        [100, 900], # BL
        [900, 100]  # TR
    ])
    ordered = page.order_corners(pts)
    assert np.allclose(ordered[0], [100, 100]) # TL
    assert np.allclose(ordered[1], [900, 100]) # TR
    assert np.allclose(ordered[2], [900, 900]) # BR
    assert np.allclose(ordered[3], [100, 900]) # BL

def test_straighten():
    img, gt_corners = create_synthetic_page()
    warped = page.straighten(img, gt_corners, "auto")
    
    h, w = warped.shape[:2]
    inset = 10
    tl = warped[inset, inset]
    br = warped[h-inset, w-inset]
    
    assert tl[0] > 200 and tl[1] > 200 and tl[2] > 200
    assert br[0] > 200 and br[1] > 200 and br[2] > 200

if __name__ == "__main__":
    test_order_corners()
    test_detect_page()
    test_straighten()
    print("Tests passed.")
