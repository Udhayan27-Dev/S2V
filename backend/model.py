import json

def build_model(geometry, solved, uf_x, uf_y, unbound_dimensions, warnings, options):
    origin_option = options.get("origin", "bottom_left")
    mirror = options.get("mirror", {})
    mirror_x = mirror.get("horizontal", False)
    mirror_y = mirror.get("vertical", False)
    
    pos_x = solved["pos_x"]
    pos_y = solved["pos_y"]
    
    const_x = {c["id"]: c["constrained"] for c in solved["classes"]["x"]}
    const_y = {c["id"]: c["constrained"] for c in solved["classes"]["y"]}
    
    # Extract vertices
    out_vertices = []
    for v in geometry.get("vertices", []):
        cx = uf_x.find(v["id"])
        cy = uf_y.find(v["id"])
        x = pos_x[cx]
        y = pos_y[cy]
        out_vertices.append({
            "id": v["id"],
            "x": x,
            "y": y,
            "estimated": not (const_x.get(cx, False) and const_y.get(cy, False))
        })
        
    # Flip Y (pixel Y down -> model Y up)
    # y = -y
    for v in out_vertices:
        v["y"] = -v["y"]
        
    if mirror_x:
        for v in out_vertices: v["x"] = -v["x"]
    if mirror_y:
        for v in out_vertices: v["y"] = -v["y"]
        
    # Bounding box
    if out_vertices:
        min_x = min(v["x"] for v in out_vertices)
        min_y = min(v["y"] for v in out_vertices)
        max_x = max(v["x"] for v in out_vertices)
        max_y = max(v["y"] for v in out_vertices)
    else:
        min_x = min_y = max_x = max_y = 0.0
        
    # Origin translation
    dx = 0.0
    dy = 0.0
    if origin_option == "bottom_left":
        dx = -min_x
        dy = -min_y
    elif origin_option == "center":
        dx = -(min_x + max_x) / 2.0
        dy = -(min_y + max_y) / 2.0
        
    for v in out_vertices:
        v["x"] = round(v["x"] + dx, 3)
        v["y"] = round(v["y"] + dy, 3)
        
    v_map = {v["id"]: v for v in out_vertices}
    
    # Extract segments
    out_segments = []
    for s in geometry.get("segments", []):
        if s["role"] in ["outline", "feature", "divider"]:
            v1 = v_map[s["v1"]]
            v2 = v_map[s["v2"]]
            length = ((v1["x"] - v2["x"])**2 + (v1["y"] - v2["y"])**2)**0.5
            out_segments.append({
                "id": s["id"],
                "v1": s["v1"],
                "v2": s["v2"],
                "role": s["role"],
                "length_mm": round(length, 3)
            })
            
    # Extract polygons
    out_polygons = []
    user_intents = options.get("user_intents", {})
    for p in geometry.get("polygons", []):
        intent = user_intents.get(p["id"], "unknown")
        # Calc area roughly
        area = 0.0
        pts = [v_map[vid] for vid in p["vertices"]]
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            area += pts[i]["x"] * pts[j]["y"] - pts[j]["x"] * pts[i]["y"]
        area = abs(area) / 2.0
        
        out_polygons.append({
            "id": p["id"],
            "vertices": p["vertices"],
            "role": p["role"],
            "intent": intent,
            "area_mm2": round(area, 3),
            "estimated": False # simplified
        })
        
    model = {
        "version": 1,
        "units": "mm",
        "origin": origin_option,
        "vertices": out_vertices,
        "segments": out_segments,
        "polygons": out_polygons,
        "constraints": solved["constraints"],
        "classes": solved["classes"],
        "estimates": solved["estimates"],
        "dof": solved["dof"],
        "conflicts": solved["conflicts"],
        "unbound_dimensions": unbound_dimensions,
        "warnings": warnings
    }
    return model
