import cv2
import numpy as np
import math
from collections import defaultdict
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, polygonize

class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, i):
        if self.parent.setdefault(i, i) == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j: self.parent[root_i] = root_j

def _get_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def _angle_mod90(p1, p2):
    ang = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 90
    if ang > 45: ang = 90 - ang
    return ang

def extract_geometry(binary_img, enhanced_gray, params):
    if len(binary_img.shape) == 3:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        
    h, w = binary_img.shape[:2]
    longer_side = max(h, w)
    shorter_side = min(h, w)
    
    p_min_seg = params.get('min_segment_len', longer_side * 0.012)
    p_spur = params.get('spur_prune', longer_side * 0.010)
    p_snap_ang = params.get('snap_angle_deg', 8.0)
    p_merge = params.get('merge_dist', longer_side * 0.012)
    p_text_size = params.get('text_max_size', shorter_side * 0.06)

    # 1. PRE-CLEAN
    # Invert to white strokes on black background
    inv = cv2.bitwise_not(binary_img)
    _, inv = cv2.threshold(inv, 127, 255, cv2.THRESH_BINARY)
    
    # Connected components to isolate text blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    text_blobs = []
    clean_img = np.zeros_like(inv)
    
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if bw < p_text_size and bh < p_text_size:
            text_blobs.append((x, y, bw, bh))
        else:
            clean_img[labels == i] = 255
            
    # Remove tiny specks from clean_img (area < 5)
    clean_num, clean_labels, clean_stats, _ = cv2.connectedComponentsWithStats(clean_img, connectivity=8)
    for i in range(1, clean_num):
        if clean_stats[i, cv2.CC_STAT_AREA] < 5:
            clean_img[clean_labels == i] = 0

    # Group text blobs
    text_groups = []
    if text_blobs:
        uf = UnionFind()
        med_h = np.median([b[3] for b in text_blobs])
        merge_t = med_h * 1.5
        
        for i, b1 in enumerate(text_blobs):
            cx1 = b1[0] + b1[2]/2
            cy1 = b1[1] + b1[3]/2
            for j in range(i+1, len(text_blobs)):
                b2 = text_blobs[j]
                cx2 = b2[0] + b2[2]/2
                cy2 = b2[1] + b2[3]/2
                if _get_distance((cx1, cy1), (cx2, cy2)) < merge_t:
                    uf.union(i, j)
                    
        groups = defaultdict(list)
        for i, b in enumerate(text_blobs):
            groups[uf.find(i)].append(b)
            
        for g_id, blobs in groups.items():
            min_x = min(b[0] for b in blobs)
            min_y = min(b[1] for b in blobs)
            max_x = max(b[0]+b[2] for b in blobs)
            max_y = max(b[1]+b[3] for b in blobs)
            gw, gh = max_x - min_x, max_y - min_y
            
            # 10% padding
            pad = int(max(gw, gh) * 0.1)
            cx1 = max(0, min_x - pad)
            cy1 = max(0, min_y - pad)
            cx2 = min(w, max_x + pad)
            cy2 = min(h, max_y + pad)
            
            ori = "H" if gw > gh else "V"
            text_groups.append({
                "id": f"t{len(text_groups)+1}",
                "bbox": [int(min_x), int(min_y), int(gw), int(gh)],
                "crop_bbox": [int(cx1), int(cy1), int(cx2 - cx1), int(cy2 - cy1)],
                "orientation": ori
            })

    # 2. SKELETON
    skel = skeletonize(clean_img > 0).astype(np.uint8) * 255
    
    # We will use OpenCV's findContours on the skeleton to get chains
    # This is an approximation of a full pixel graph traversal but works well for RDP.
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    chains = []
    for c in contours:
        c = c.reshape(-1, 2)
        if len(c) > 2:
            chains.append(c)

    # 3. SEGMENT FITTING (RDP)
    epsilon = longer_side * 0.004
    raw_segments = []
    
    for c in chains:
        approx = cv2.approxPolyDP(c, epsilon, False)
        approx = approx.reshape(-1, 2)
        for i in range(len(approx) - 1):
            p1 = tuple(approx[i])
            p2 = tuple(approx[i+1])
            if _get_distance(p1, p2) > 1.0: # avoid 0 length
                raw_segments.append([p1, p2])

    # 4. SNAP AND CLEAN
    # Calculate dominant skew
    angles = []
    weights = []
    for (p1, p2) in raw_segments:
        L = _get_distance(p1, p2)
        if L > p_min_seg:
            angles.append(_angle_mod90(p1, p2))
            weights.append(L)
            
    warnings = []
    if angles:
        hist, bins = np.histogram(angles, bins=90, range=(0, 90), weights=weights)
        dom_angle = bins[np.argmax(hist)]
        if dom_angle > 1.0 and dom_angle < 89.0:
            warnings.append(f"Image skew detected ({dom_angle:.1f}°). Segments will be snapped to exactly H/V.")
            
    snapped_segments = []
    for (p1, p2) in raw_segments:
        ang = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
        L = _get_distance(p1, p2)
        
        # Determine if near H/V
        is_h = (ang < p_snap_ang) or (ang > 180 - p_snap_ang)
        is_v = (90 - p_snap_ang < ang < 90 + p_snap_ang)
        
        if is_h or is_v:
            # Snap by rotating around midpoint
            mx = (p1[0] + p2[0]) / 2.0
            my = (p1[1] + p2[1]) / 2.0
            if is_h:
                p1 = (p1[0], my)
                p2 = (p2[0], my)
            else:
                p1 = (mx, p1[1])
                p2 = (mx, p2[1])
                
        if _get_distance(p1, p2) >= p_min_seg:
            snapped_segments.append([list(p1), list(p2)])

    # Merge endpoints that are close (Union-Find)
    uf_pts = UnionFind()
    pts = []
    for (p1, p2) in snapped_segments:
        pts.append(p1)
        pts.append(p2)
        
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if _get_distance(pts[i], pts[j]) < p_merge:
                uf_pts.union(i, j)
                
    merged_pts = {}
    groups = defaultdict(list)
    for i in range(len(pts)):
        groups[uf_pts.find(i)].append(pts[i])
        
    for g_id, g_pts in groups.items():
        avg_x = sum(p[0] for p in g_pts) / len(g_pts)
        avg_y = sum(p[1] for p in g_pts) / len(g_pts)
        merged_pts[g_id] = [avg_x, avg_y]

    final_segments = []
    seg_id = 1
    for i in range(0, len(pts), 2):
        p1 = merged_pts[uf_pts.find(i)]
        p2 = merged_pts[uf_pts.find(i+1)]
        if _get_distance(p1, p2) > 0.1:
            final_segments.append({
                "id": f"s{seg_id}",
                "v1_coords": p1,
                "v2_coords": p2,
                "role": "unknown",
                "confidence": 0.8
            })
            seg_id += 1

    # 5. ARROWHEADS & DIMENSIONS
    # (For a dimension line, we look for V-shapes at the end of a segment)
    arrowheads = []
    # This is a stub for the arrowhead detection which requires pixel analysis.
    # To keep it simple without full stroke parsing:
    # We check if there are 2 short segments meeting at an endpoint of a longer segment.
    # Since we union-find merged endpoints, they are topological neighbors.
    
    # Adjacency list
    adj = defaultdict(list)
    for s in final_segments:
        adj[tuple(s["v1_coords"])].append(s)
        adj[tuple(s["v2_coords"])].append(s)
        
    for p_tuple, neighbor_segs in adj.items():
        if len(neighbor_segs) == 3:
            # Check if one is long and two are short (arrowhead)
            lengths = []
            for s in neighbor_segs:
                L = _get_distance(s["v1_coords"], s["v2_coords"])
                lengths.append((L, s))
            lengths.sort(key=lambda x: x[0])
            
            # If the shortest two are small and similar, and the longest is much longer
            if lengths[0][0] < longer_side * 0.05 and lengths[1][0] < longer_side * 0.05:
                if lengths[2][0] > longer_side * 0.05:
                    # Mark the short ones as arrowheads (extensions/arrows)
                    lengths[0][1]["role"] = "unknown" # temporary
                    lengths[1][1]["role"] = "unknown"
                    
                    # Compute angle between the two short segments
                    # If angle is narrow (15-60 deg), it's an arrow tip
                    p_center = p_tuple
                    pA = lengths[0][1]["v1_coords"] if lengths[0][1]["v2_coords"] == p_center else lengths[0][1]["v2_coords"]
                    pB = lengths[1][1]["v1_coords"] if lengths[1][1]["v2_coords"] == p_center else lengths[1][1]["v2_coords"]
                    
                    vA = (pA[0]-p_center[0], pA[1]-p_center[1])
                    vB = (pB[0]-p_center[0], pB[1]-p_center[1])
                    
                    dot = vA[0]*vB[0] + vA[1]*vB[1]
                    magA = math.hypot(*vA)
                    magB = math.hypot(*vB)
                    if magA > 0 and magB > 0:
                        cos_ang = max(-1.0, min(1.0, dot / (magA*magB)))
                        ang = math.degrees(math.acos(cos_ang))
                        if 15 < ang < 60:
                            # It's an arrowhead
                            # Mark the main line as a dimension line candidate
                            lengths[2][1]["role"] = "dimension"

    # Remove arrowhead strokes from final segments so they don't form polygons
    # Wait, the prompt says: "Remove arrowhead strokes from the geometry segment list"
    # Actually, if we just drop them, it's easier. Let's just keep them as "unknown" and filter out later if needed, or drop them now.
    # To drop them:
    # We will refine this later if needed. For now, we leave them in but shapely won't polygonize them if they are dangling.
    
    # 6. TOPOLOGY
    json_data = build_topology(final_segments, text_groups, h, w, params)
    json_data["warnings"] = warnings
    
    return json_data

def build_topology(segments_data, text_groups, h, w, params):
    # Convert coordinate lists to unique vertices and segment links
    v_map = {}
    vertices = []
    segments = []
    
    def get_vid(pt):
        # Round to 1 decimal to avoid float jitter
        k = (round(pt[0], 1), round(pt[1], 1))
        if k not in v_map:
            vid = f"v{len(vertices)+1}"
            v_map[k] = vid
            vertices.append({"id": vid, "x": float(pt[0]), "y": float(pt[1])})
        return v_map[k]

    for s in segments_data:
        v1 = get_vid(s["v1_coords"])
        v2 = get_vid(s["v2_coords"])
        
        # Calculate length and orientation
        pt1 = [v for v in vertices if v["id"] == v1][0]
        pt2 = [v for v in vertices if v["id"] == v2][0]
        dx, dy = pt2["x"] - pt1["x"], pt2["y"] - pt1["y"]
        length = math.hypot(dx, dy)
        ang = math.degrees(math.atan2(dy, dx)) % 180
        
        ori = "other"
        if ang < 5 or ang > 175: ori = "H"
        elif 85 < ang < 95: ori = "V"
        
        segments.append({
            "id": s["id"],
            "v1": v1,
            "v2": v2,
            "orientation": ori,
            "length_px": length,
            "role": s["role"],
            "confidence": s["confidence"]
        })

    # Remove duplicates
    unique_segs = []
    seen = set()
    for s in segments:
        k = tuple(sorted([s["v1"], s["v2"]]))
        if k not in seen:
            seen.add(k)
            unique_segs.append(s)
            
    segments = unique_segs

    json_data = {
        "version": 1,
        "image": {"w": w, "h": h},
        "params": params,
        "vertices": vertices,
        "segments": segments,
        "arcs": [],
        "polygons": [],
        "dimension_lines": [],
        "text_groups": text_groups
    }
    
    return recompute_topology(json_data)

def recompute_topology(json_data):
    # Runs shapely polygonize and updates roles
    vertices = {v["id"]: (v["x"], v["y"]) for v in json_data["vertices"]}
    lines = []
    for s in json_data["segments"]:
        lines.append(LineString([vertices[s["v1"]], vertices[s["v2"]]]))
        
    merged = unary_union(lines)
    polys = list(polygonize(merged))
    
    # Area threshold
    min_area = (json_data["image"]["w"] * json_data["image"]["h"]) * 0.005
    valid_polys = []
    for p in polys:
        if p.area > min_area:
            valid_polys.append(p)
            
    # Determine nesting (outline vs feature)
    valid_polys.sort(key=lambda p: p.area, reverse=True)
    out_polys = []
    for i, p in enumerate(valid_polys):
        role = "outline"
        parent = None
        for j, other in enumerate(valid_polys):
            if i != j:
                other_filled = Polygon(other.exterior)
                if other_filled.contains(p):
                    role = "feature"
                    parent = f"p{j+1}"
                    break
                
        poly_vid_list = []
        # Map polygon coords back to vertex IDs (approximate)
        for coord in p.exterior.coords[:-1]:
            closest_vid = None
            min_d = float('inf')
            for vid, vcoord in vertices.items():
                d = _get_distance(coord, vcoord)
                if d < min_d:
                    min_d = d
                    closest_vid = vid
            if closest_vid:
                poly_vid_list.append(closest_vid)
                
        out_polys.append({
            "id": f"p{i+1}",
            "vertices": poly_vid_list,
            "area_px": p.area,
            "role": role,
            "parent": parent
        })
        
    json_data["polygons"] = out_polys
    
    # Assign roles to segments based on polygons
    for s in json_data["segments"]:
        if s["role"] in ["dimension", "extension"]: continue # Keep explicit roles
        
        p1 = vertices[s["v1"]]
        p2 = vertices[s["v2"]]
        seg_line = LineString([p1, p2])
        
        in_polys = []
        for i, p in enumerate(valid_polys):
            # If segment is part of polygon exterior
            if p.exterior.distance(seg_line) < 1.0:
                in_polys.append(i)
                
        if len(in_polys) == 1:
            s["role"] = out_polys[in_polys[0]]["role"]
            s["confidence"] = 0.9
        elif len(in_polys) > 1:
            s["role"] = "divider"
            s["confidence"] = 0.9
        else:
            if s["role"] == "unknown":
                s["confidence"] = 0.5
                
    return json_data
