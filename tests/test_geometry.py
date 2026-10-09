import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
import geometry

def test_extract_geometry():
    # Create synthetic L-shape with dimension lines
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    
    # Draw L shape (thick strokes to simulate marker/pencil)
    # 24x10 and 12x10 -> let's say 400x200 and 200x200
    pts = np.array([
        [200, 200], [600, 200], [600, 400], [400, 400], [400, 600], [200, 600]
    ], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(50, 50, 50), thickness=4)
    
    # Draw a divider
    cv2.line(img, (200, 400), (400, 400), (50, 50, 50), thickness=4)
    
    # Draw an inner feature (rectangle)
    f_pts = np.array([
        [250, 250], [350, 250], [350, 350], [250, 350]
    ], dtype=np.int32)
    cv2.polylines(img, [f_pts], isClosed=True, color=(50, 50, 50), thickness=3)
    
    # Draw dimension line below
    cv2.line(img, (200, 700), (600, 700), (50, 50, 50), thickness=2)
    # arrow heads
    cv2.line(img, (200, 700), (220, 680), (50, 50, 50), thickness=2)
    cv2.line(img, (200, 700), (220, 720), (50, 50, 50), thickness=2)
    cv2.line(img, (600, 700), (580, 680), (50, 50, 50), thickness=2)
    cv2.line(img, (600, 700), (580, 720), (50, 50, 50), thickness=2)
    
    # Text blob
    cv2.putText(img, "400", (380, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    # Binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    params = {}
    json_data = geometry.extract_geometry(binary, gray, params)
    
    # Assertions
    assert "vertices" in json_data
    assert "segments" in json_data
    assert "polygons" in json_data
    
    # We should have found polygons
    import pprint
    pprint.pprint(json_data)
    
    assert len(json_data["polygons"]) > 0
    
    roles = [s["role"] for s in json_data["segments"]]
    assert "outline" in roles
    assert "feature" in roles
    
    print("Geometry extraction tests passed.")

if __name__ == "__main__":
    test_extract_geometry()
