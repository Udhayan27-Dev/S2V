import numpy as np
from collections import defaultdict
import uuid

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, i):
        if self.parent.setdefault(i, i) == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def build_classes(geometry, tol=10.0):
    """Build X and Y coordinate classes from geometry."""
    uf_x = UnionFind()
    uf_y = UnionFind()
    
    vertices = {v["id"]: v for v in geometry.get("vertices", [])}
    
    # Base classes from orthogonal segments
    for s in geometry.get("segments", []):
        if s["role"] in ["outline", "feature", "divider", "dimension", "extension"]:
            if s["orientation"] == "V":
                uf_x.union(s["v1"], s["v2"])
            elif s["orientation"] == "H":
                uf_y.union(s["v1"], s["v2"])
                
    # Infer additional alignments
    inferred = []
    
    # Group vertices by class to find average pixel coordinate
    x_class_pts = defaultdict(list)
    y_class_pts = defaultdict(list)
    
    for vid, v in vertices.items():
        x_class_pts[uf_x.find(vid)].append(v["x"])
        y_class_pts[uf_y.find(vid)].append(v["y"])
        
    x_class_pos = {cid: np.mean(pts) for cid, pts in x_class_pts.items()}
    y_class_pos = {cid: np.mean(pts) for cid, pts in y_class_pts.items()}
    
    x_classes = list(x_class_pos.keys())
    for i in range(len(x_classes)):
        for j in range(i+1, len(x_classes)):
            c1, c2 = x_classes[i], x_classes[j]
            if abs(x_class_pos[c1] - x_class_pos[c2]) < tol:
                inferred.append({
                    "id": f"inf_{uuid.uuid4().hex[:8]}",
                    "axis": "x",
                    "c1": c1, "c2": c2,
                    "diff_px": abs(x_class_pos[c1] - x_class_pos[c2])
                })
                
    y_classes = list(y_class_pos.keys())
    for i in range(len(y_classes)):
        for j in range(i+1, len(y_classes)):
            c1, c2 = y_classes[i], y_classes[j]
            if abs(y_class_pos[c1] - y_class_pos[c2]) < tol:
                inferred.append({
                    "id": f"inf_{uuid.uuid4().hex[:8]}",
                    "axis": "y",
                    "c1": c1, "c2": c2,
                    "diff_px": abs(y_class_pos[c1] - y_class_pos[c2])
                })
                
    return uf_x, uf_y, inferred

def build_constraints(geometry, dimensions_data, uf_x, uf_y, options):
    """Build distance constraints from dimensions."""
    constraints = []
    unbound = []
    
    accepted_inferred = options.get("accept_inferred", [])
    for inf in accepted_inferred:
        # If accepted, we merge the classes. But since we need to track it as a constraint,
        # we can just add a 0-distance constraint.
        constraints.append({
            "id": inf["id"],
            "type": "distance",
            "axis": inf["axis"],
            "a": inf["c1"],
            "b": inf["c2"],
            "value_mm": 0.0,
            "dimension": None,
            "source": "inferred",
            "text": "0",
            "weight": 10.0
        })
        
    vertices = {v["id"]: v for v in geometry.get("vertices", [])}
    segments = {s["id"]: s for s in geometry.get("segments", [])}
    
    for dim in dimensions_data.get("dimensions", []):
        if dim.get("status") not in ["ok", "check"] and not dim.get("user_edited"):
            if dim.get("status") != "manual":
                continue
                
        dim_line_id = dim.get("dimension_line")
        dl = segments.get(dim_line_id)
        if not dl:
            unbound.append({"dimension": dim["id"], "reason": "No dimension line found"})
            continue
            
        axis = "x" if dl["orientation"] == "H" else "y" if dl["orientation"] == "V" else None
        if not axis:
            unbound.append({"dimension": dim["id"], "reason": "Dimension line is not orthogonal"})
            continue
            
        uf = uf_x if axis == "x" else uf_y
        
        c1, c2 = None, None
        
        # If there are measured segments, try to use their classes
        ms = dim.get("measured_segments", [])
        if len(ms) >= 2:
            s1 = segments.get(ms[0])
            s2 = segments.get(ms[-1])
            if s1 and s2:
                # For H dimension line (X axis distance), the measured features are usually V lines (Y axis constant, X varies between them)
                # We need the X class of s1 and s2
                c1 = uf.find(s1["v1"])
                c2 = uf.find(s2["v1"])
        
        if c1 is None or c2 is None or c1 == c2:
            # Fallback to dimension line endpoints
            # The dimension line endpoints might be connected to extension lines.
            c1 = uf.find(dl["v1"])
            c2 = uf.find(dl["v2"])
            
        if c1 == c2:
            unbound.append({"dimension": dim["id"], "reason": "Resolved to same coordinate class"})
            continue
            
        # Determine sign: b should be > a in pixel coordinates
        v1 = vertices[dl["v1"]]
        v2 = vertices[dl["v2"]]
        
        p1 = v1["x"] if axis == "x" else v1["y"]
        p2 = v2["x"] if axis == "x" else v2["y"]
        
        # We want (b - a = value). So b is the one with larger coordinate.
        # But wait, pixel Y is down. In pixel coords, larger Y means lower on page.
        # We will flip Y globally later, so for now, we just enforce b > a in pixel space,
        # which means for Y, b is lower. After flip (Y = H - Y), b will be lower Y, meaning a is higher.
        # Wait, if b > a in pixel Y, then after Y' = -Y, b' < a'.
        # So b' - a' = -value!
        # To fix this, for Y axis, we want the constraint in the model (y up) to be positive.
        # If axis == 'y', in pixel space, if p2 > p1 (p2 is below p1), then in model space p2 < p1.
        # So we should swap so that a is the lower (bottom) one in model space, which is the HIGHER in pixel space.
        
        if axis == "x":
            if p2 > p1:
                a, b = c1, c2
            else:
                a, b = c2, c1
        else: # axis == "y"
            # Model space Y is UP. Pixel space Y is DOWN.
            # We want b(model) > a(model) => b(pixel) < a(pixel).
            # So a should be the one with LARGER pixel Y.
            if p2 < p1:
                a, b = c1, c2
            else:
                a, b = c2, c1
                
        weight = 10.0 if dim.get("user_edited") else 1.0
        
        constraints.append({
            "id": f"c_{dim['id']}",
            "type": "distance",
            "axis": axis,
            "a": a,
            "b": b,
            "value_mm": dim["value_mm"],
            "dimension": dim["id"],
            "source": "user" if dim.get("user_edited") else "read",
            "text": dim.get("raw_text", ""),
            "weight": weight
        })
        
    # Also add user custom edge constraints
    for uc in options.get("user_constraints", []):
        constraints.append(uc)
        
    return constraints, unbound
