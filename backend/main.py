import base64
import time
import json
from io import BytesIO
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import uvicorn
import os

from enhance import enhance
from quality import check_quality
import page

app = FastAPI()

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_base64(cv2_image):
    _, buffer = cv2.imencode('.png', cv2_image)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def load_image(contents):
    pil_image = Image.open(BytesIO(contents))
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/detect-page")
async def detect_page_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = load_image(contents)
        
        orig_w, orig_h = pil_image.size
        img_bgr = pil_to_cv2(pil_image)
        
        # Detect corners
        corners, confidence, method, warnings = page.detect_page(img_bgr)
        
        # Create preview
        preview_max = 1600.0
        scale = 1.0
        if max(orig_w, orig_h) > preview_max:
            scale = preview_max / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pil_preview = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            pil_preview = pil_image
            
        preview_cv2 = pil_to_cv2(pil_preview)
        preview_b64 = cv2_to_base64(preview_cv2)
        
        # Scale corners for UI
        preview_corners = [[x * scale, y * scale] for [x, y] in corners]
        
        return JSONResponse({
            "preview": preview_b64,
            "scale": scale,
            "corners_orig": corners,
            "corners_preview": preview_corners,
            "confidence": confidence,
            "method": method,
            "warnings": warnings,
            "original_size": [orig_w, orig_h]
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/straighten")
async def straighten_api(
    file: UploadFile = File(...),
    corners: str = Form(...),
    paper: str = Form("auto"),
    rotate: int = Form(0)
):
    try:
        contents = await file.read()
        pil_image = load_image(contents)
        img_bgr = pil_to_cv2(pil_image)
        
        pts = json.loads(corners)
        if len(pts) != 4:
            raise ValueError("Corners must be exactly 4 points")
            
        # Validate convex quad
        pts_np = np.array(pts, dtype="float32")
        if not cv2.isContourConvex(pts_np.reshape(-1, 1, 2)):
            return JSONResponse(status_code=400, content={"error": "Corners do not form a valid convex shape. Please adjust."})
            
        warped = page.straighten(img_bgr, pts_np, paper)
        if rotate > 0:
            warped = page.rotate(warped, rotate)
            
        h, w = warped.shape[:2]
        warped_b64 = cv2_to_base64(warped)
        
        return JSONResponse({
            "straightened": warped_b64,
            "size": [w, h]
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/enhance")
async def process_image(
    file: UploadFile = File(...),
    black_pct: float = Form(82.0),
    gamma: float = Form(0.9),
    stroke_px: int = Form(None)
):
    try:
        start_time = time.time()
        
        contents = await file.read()
        pil_image = load_image(contents)
            
        w, h = pil_image.size
        longer_side = max(w, h)
        
        if longer_side > 4000:
            scale = 4000.0 / longer_side
            new_w, new_h = int(w * scale), int(h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
            
        preview_size = (w, h)
        if max(w, h) > 800:
            scale = 800.0 / max(w, h)
            preview_size = (int(w * scale), int(h * scale))
        preview_image = pil_image.copy()
        preview_image.thumbnail(preview_size)
        preview_cv2 = pil_to_cv2(preview_image)
        orig_b64 = cv2_to_base64(preview_cv2)
        
        img_bgr = pil_to_cv2(pil_image)
        enhanced_gray, binary = enhance(
            img_bgr, 
            black_pct=black_pct, 
            gamma=gamma, 
            stroke_px=stroke_px
        )
        
        warnings = check_quality(img_bgr, binary)
        
        enhanced_b64 = cv2_to_base64(enhanced_gray)
        binary_b64 = cv2_to_base64(binary)
        
        end_time = time.time()
        
        return JSONResponse({
            "original": orig_b64,
            "enhanced": enhanced_b64,
            "binary": binary_b64,
            "warnings": warnings,
            "image_size": [w, h],
            "processing_time_s": round(end_time - start_time, 2)
        })
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

import geometry

@app.post("/api/geometry")
async def geometry_api(
    binary: UploadFile = File(...),
    enhanced: UploadFile = File(None),
    min_segment_len: float = Form(None),
    spur_prune: float = Form(None),
    snap_angle_deg: float = Form(8.0),
    merge_dist: float = Form(None),
    text_max_size: float = Form(None)
):
    try:
        bin_bytes = await binary.read()
        bin_img = pil_to_cv2(load_image(bin_bytes))
        
        enh_img = None
        if enhanced:
            enh_bytes = await enhanced.read()
            enh_img = pil_to_cv2(load_image(enh_bytes))
        
        # Build params
        h, w = bin_img.shape[:2]
        long_side = max(h, w)
        short_side = min(h, w)
        params = {
            "min_segment_len": min_segment_len if min_segment_len else long_side * 0.012,
            "spur_prune": spur_prune if spur_prune else long_side * 0.01,
            "snap_angle_deg": snap_angle_deg,
            "merge_dist": merge_dist if merge_dist else long_side * 0.012,
            "text_max_size": text_max_size if text_max_size else short_side * 0.06
        }
        
        json_data = geometry.extract_geometry(bin_img, enh_img, params)
        
        return JSONResponse(json_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/geometry/overlay")
async def geometry_overlay_api(
    geometry_json: str = Form(...),
    enhanced: UploadFile = File(None)
):
    try:
        data = json.loads(geometry_json)
        updated_data = geometry.recompute_topology(data)
        return JSONResponse(updated_data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
import dimensions
from units import to_mm

@app.post("/api/dimensions")
async def dimensions_api(
    geometry_json: str = Form(...),
    enhanced: UploadFile = File(...),
    engine: str = Form("easyocr"),
    default_unit: str = Form("mm")
):
    try:
        geom = json.loads(geometry_json)
        enh_bytes = await enhanced.read()
        enh_img = pil_to_cv2(load_image(enh_bytes))
        if len(enh_img.shape) == 3:
            enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2GRAY)
            
        options = {
            "engine": engine,
            "default_unit": default_unit
        }
        
        result = dimensions.extract_dimensions(geom, enh_img, options)
        
        # Add crops to the result for UI
        crops = {}
        for tg in geom.get("text_groups", []):
            crop = dimensions.get_crop(enh_img, tg["crop_bbox"])
            crops[tg["id"]] = cv2_to_base64(crop)
            
        result["crops"] = crops
        return JSONResponse(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/dimensions/parse")
async def dimensions_parse_api(
    text: str = Form(...),
    default_unit: str = Form("mm")
):
    try:
        parsed = to_mm(text, default_unit)
        if parsed:
            return JSONResponse({"status": "ok", "parsed": parsed})
        return JSONResponse({"status": "error", "error": "Could not parse value"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/dimensions/recompute")
async def dimensions_recompute_api(
    dimensions_json: str = Form(...)
):
    try:
        # Not fully re-running OCR, just re-evaluating sheet context based on edited dimensions
        data = json.loads(dimensions_json)
        # We can just run the sheet evaluation part
        scales = []
        units_seen = set()
        warnings = []
        
        for d in data.get("dimensions", []):
            if d.get("status") != "ignored":
                units_seen.add(d["unit_used"])
                # We don't have segment lengths in this minimal JSON unless passed. 
                # Assuming the frontend just wants warnings updated.
                
        if len(units_seen) > 1:
            warnings.append(f"Mixed units detected on sheet: {', '.join(units_seen)}")
            
        vals = [d["value_mm"] for d in data.get("dimensions", []) if d["value_mm"] > 0 and d.get("status") != "ignored"]
        if vals and (max(vals) / min(vals)) > 1000:
            warnings.append("Values differ by >1000x. Check for unit confusion.")
            
        data["warnings"] = warnings
        data["sheet"]["units_seen"] = list(units_seen)
        
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
from fastapi import Request
import constraints
import solver
import model

@app.post("/api/solve")
async def solve_api(request: Request):
    try:
        body = await request.json()
        geometry = body.get("geometry", {})
        dimensions_data = body.get("dimensions", {})
        options = body.get("options", {})
        
        uf_x, uf_y, inferred = constraints.build_classes(geometry)
        
        # Add inferred to options if the frontend didn't pass accept_inferred
        if "inferred_alignments" not in options:
            options["inferred_alignments"] = inferred
            
        constrs, unbound = constraints.build_constraints(geometry, dimensions_data, uf_x, uf_y, options)
        
        constrs_x = [c for c in constrs if c["axis"] == "x"]
        constrs_y = [c for c in constrs if c["axis"] == "y"]
        
        solved = solver.solve(geometry, constrs_x, constrs_y, uf_x, uf_y, options)
        
        mod = model.build_model(geometry, solved, uf_x, uf_y, unbound, dimensions_data.get("warnings", []), options)
        
        # Attach inferred alignments so the UI can prompt the user
        mod["inferred_alignments"] = inferred
        
        return JSONResponse(mod)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/solve/edit")
async def solve_edit_api(request: Request):
    try:
        body = await request.json()
        # To make it stateless, we just expect the frontend to pass the updated options/geometry/dimensions
        # and we re-run the whole solver. This is fast (<200ms).
        geometry = body.get("geometry", {})
        dimensions_data = body.get("dimensions", {})
        options = body.get("options", {})
        
        uf_x, uf_y, inferred = constraints.build_classes(geometry)
        constrs, unbound = constraints.build_constraints(geometry, dimensions_data, uf_x, uf_y, options)
        
        constrs_x = [c for c in constrs if c["axis"] == "x"]
        constrs_y = [c for c in constrs if c["axis"] == "y"]
        
        solved = solver.solve(geometry, constrs_x, constrs_y, uf_x, uf_y, options)
        mod = model.build_model(geometry, solved, uf_x, uf_y, unbound, dimensions_data.get("warnings", []), options)
        mod["inferred_alignments"] = inferred
        return JSONResponse(mod)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

os.makedirs(os.path.join(os.path.dirname(__file__), "../frontend"), exist_ok=True)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../frontend"), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
