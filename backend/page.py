import cv2
import numpy as np
import math

def order_corners(pts):
    """
    Order points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def score_quad(quad, img_shape):
    """
    Score a quadrilateral based on convexity, area, angle to 90 deg.
    """
    h, w = img_shape[:2]
    img_area = h * w
    
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return 0.0

    area = cv2.contourArea(quad)
    area_ratio = area / img_area
    if area_ratio < 0.25:
        return 0.0

    def angle_between(v1, v2):
        dot = np.dot(v1, v2)
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.atan2(det, dot)
        return abs(math.degrees(angle))
        
    pts = quad.reshape(4, 2)
    angles = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        p3 = pts[(i+2)%4]
        v1 = p1 - p2
        v2 = p3 - p2
        ang = angle_between(v1, v2)
        angles.append(ang)
        
    angle_dev = sum([abs(90 - a) for a in angles]) / 4.0
    angle_score = max(0, 1.0 - (angle_dev / 45.0))
    area_score = area_ratio
    
    return (angle_score * 0.5) + (area_score * 0.5)

def detect_page(img_bgr):
    """
    Detects the page in the image.
    Returns: corners (TL, TR, BR, BL in original scale), confidence (0-1), method, warnings.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    
    # Scale down for speed
    max_side = 1000.0
    scale = 1.0
    if max(orig_h, orig_w) > max_side:
        scale = max_side / max(orig_h, orig_w)
        
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_small = cv2.resize(img_bgr, (new_w, new_h))
    
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    best_quad = None
    best_score = 0.0
    best_method = ""
    
    # Method A: Edge Detection
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            score = score_quad(approx, img_small.shape)
            if score > best_score:
                best_score = score
                best_quad = approx
                best_method = "edge"
                
    # Method B: Brightness/color (Otsu)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            score = score_quad(approx, img_small.shape)
            if score > best_score:
                best_score = score
                best_quad = approx
                best_method = "brightness"
                
    # Method C: GrabCut
    if best_score < 0.6:
        mask = np.zeros(img_small.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        margin = int(min(new_w, new_h) * 0.1)
        rect = (margin, margin, new_w - 2*margin, new_h - 2*margin)
        try:
            cv2.grabCut(img_small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            thresh_gc = mask2 * 255
            contours, _ = cv2.findContours(thresh_gc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    score = score_quad(approx, img_small.shape)
                    if score > best_score:
                        best_score = score
                        best_quad = approx
                        best_method = "grabcut"
        except:
            pass

    warnings = []
    
    if best_quad is None or best_score < 0.3:
        best_method = "fallback"
        best_score = 0.0
        warnings.append("Page not detected; drag the corners to fit the paper.")
        margin_w = int(orig_w * 0.05)
        margin_h = int(orig_h * 0.05)
        corners = np.array([
            [margin_w, margin_h],
            [orig_w - margin_w, margin_h],
            [orig_w - margin_w, orig_h - margin_h],
            [margin_w, orig_h - margin_h]
        ], dtype="float32")
    else:
        corners = best_quad.reshape(4, 2) / scale
        margin = max(orig_w, orig_h) * 0.01
        for pt in corners:
            if pt[0] <= margin or pt[0] >= orig_w - margin or pt[1] <= margin or pt[1] >= orig_h - margin:
                if "One or more corners appear to be cut off by the frame." not in warnings:
                    warnings.append("One or more corners appear to be cut off by the frame.")
                    break

    corners = order_corners(corners)
    np.clip(corners[:, 0], 0, orig_w - 1, out=corners[:, 0])
    np.clip(corners[:, 1], 0, orig_h - 1, out=corners[:, 1])

    return corners.tolist(), float(best_score), best_method, warnings

def straighten(img_bgr, corners, paper="auto"):
    """
    Warps the image based on the 4 corners.
    """
    rect = order_corners(np.array(corners, dtype="float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    if paper != "auto":
        is_portrait = maxHeight > maxWidth
        ratio = 1.414 if paper == "a4" else 1.294
        
        if is_portrait:
            maxWidth = int(maxHeight / ratio)
        else:
            maxHeight = int(maxWidth / ratio)

    # Inward margin 0.5%
    margin_w = maxWidth * 0.005
    margin_h = maxHeight * 0.005

    dst = np.array([
        [-margin_w, -margin_h],
        [maxWidth - 1 + margin_w, -margin_h],
        [maxWidth - 1 + margin_w, maxHeight - 1 + margin_h],
        [-margin_w, maxHeight - 1 + margin_h]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    
    h, w = warped.shape[:2]
    if max(h, w) > 4000:
        scale = 4000.0 / max(h, w)
        warped = cv2.resize(warped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    return warped

def rotate(img_bgr, quarter_turns):
    turns = quarter_turns % 4
    if turns == 1:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif turns == 2:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif turns == 3:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr
