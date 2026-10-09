import re
import numpy as np
import math
from collections import defaultdict
from ocr import recognize
from units import to_mm
import cv2

def clean_text(text):
    original = text
    # Only applies in digit context, meaning if it looks like a confused number
    # O/o/D->0, l/I/|/i->1, Z/z->2, S/s->5, B->8, g/q->9
    
    # 'nm'/'rn'/'mrn'/'mm.'->'mm'
    text = re.sub(r'\b(nm|rn|rnm|mrn|mm\.)\b', 'mm', text, flags=re.IGNORECASE)
    
    # '' -> "
    text = text.replace("''", '"')
    
    # common digit confusions using regex
    # Replace O,o,D with 0 if surrounded by digits or at end/start of digit string
    # Actually, simple translation table for specific characters inside numbers
    # A bit complex to do safely, but let's do targeted replaces
    
    parts = []
    corrected = False
    for token in text.split():
        # if the token has digits, apply digit cleaning
        if any(c.isdigit() for c in token) or len(token) <= 2:
            clean_token = token
            # S -> 5
            if 'S' in clean_token or 's' in clean_token:
                if len(clean_token) == 1 or any(c.isdigit() for c in clean_token):
                    clean_token = clean_token.replace('S', '5').replace('s', '5')
            # O, o, D -> 0
            if 'O' in clean_token or 'o' in clean_token:
                # only if not 'mm', 'cm', 'm' etc
                if clean_token.lower() not in ['mm', 'cm', 'm', 'in', 'ft']:
                    clean_token = clean_token.replace('O', '0').replace('o', '0')
            # l, I, |, i -> 1
            if 'l' in clean_token or 'I' in clean_token or '|' in clean_token or 'i' in clean_token:
                if clean_token.lower() not in ['in']:
                    clean_token = clean_token.replace('l', '1').replace('I', '1').replace('|', '1').replace('i', '1')
            # Z, z -> 2
            if 'Z' in clean_token or 'z' in clean_token:
                clean_token = clean_token.replace('Z', '2').replace('z', '2')
            # B -> 8
            if 'B' in clean_token:
                clean_token = clean_token.replace('B', '8')
            # g, q -> 9
            if 'g' in clean_token or 'q' in clean_token:
                clean_token = clean_token.replace('g', '9').replace('q', '9')
            
            if clean_token != token:
                corrected = True
            parts.append(clean_token)
        else:
            parts.append(token)
            
    res = " ".join(parts)
    # Check if ' -> ' (feet marks)
    res = res.replace("' ", "'")
    return res, corrected

def _get_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def get_crop(img_gray, bbox):
    # bbox: min_x, min_y, w, h
    x, y, w, h = bbox
    return img_gray[y:y+h, x:x+w]

import time
import hashlib

_ocr_cache = {}

def process_text_group(tg, crop_gray, default_unit="mm", engine="trocr-small"):
    # Hash for caching
    crop_hash = hashlib.sha256(crop_gray.tobytes()).hexdigest()
    cache_key = f"{crop_hash}_{engine}_{default_unit}"
    if cache_key in _ocr_cache:
        return _ocr_cache[cache_key]

    ori = tg.get("orientation", "H")
    
    start_time = time.time()
    candidate_generator = recognize(crop_gray, ori, engine)
    
    scored_candidates = []
    
    # We evaluate one batch at a time
    for candidate_batch in candidate_generator:
        # candidate_batch is just a single candidate in our new yield logic, but the generator actually yields candidate dictionaries
        # Wait, the generator yields the candidate directly. Let's treat it as a single item.
        # Actually my generator yields `cand` one by one! Let me just iterate over it.
        # BUT ocr_trocr yields results_for_cand (a list of up to 3 beams) per image. Let me handle both.
        if isinstance(candidate_batch, dict):
            batch = [candidate_batch]
        else:
            batch = candidate_batch
            
        batch_scored = []
        for c in batch:
            raw_text = c["text"]
            conf = c["confidence"]
            
            cleaned_text, corrected = clean_text(raw_text)
            parsed = to_mm(cleaned_text, default_unit)
            
            if parsed:
                score = conf
                if corrected: score -= 0.1
                if not parsed["assumed"]: score += 0.2
                if 0.1 <= parsed["value_mm"] <= 100000:
                    score += 0.1
                else:
                    score -= 0.3
                    
                score = min(max(score, 0.0), 1.0)
                
                # Check consistency across rotations
                # We'll just trust the score for now
                cand_res = {
                    "text": cleaned_text,
                    "raw_text": raw_text,
                    "value_mm": parsed["value_mm"],
                    "unit_used": parsed["unit_used"],
                    "kind": parsed["kind"],
                    "assumed_unit": parsed["assumed"],
                    "normalized_text": parsed["normalized_text"],
                    "score": score,
                    "variant": c.get("variant", ""),
                    "rotation": c.get("rotation", 0),
                    "confidence": conf
                }
                batch_scored.append(cand_res)
                scored_candidates.append(cand_res)
                
        # Early stopping if we found a highly confident and valid parse in this batch
        if batch_scored:
            best_in_batch = max(batch_scored, key=lambda x: x["score"])
            if best_in_batch["score"] >= 0.8:
                break
                
    elapsed_time = time.time() - start_time
                
    # Sort all accumulated candidates and take top 3
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate by text
    seen = set()
    deduped = []
    for c in scored_candidates:
        if c["text"] not in seen:
            seen.add(c["text"])
            deduped.append(c)
            
    best = deduped[:3]
    
    if not best:
        res = {
            "status": "unreadable",
            "best": None,
            "alternatives": [],
            "process_time": elapsed_time
        }
    else:
        top = best[0]
        alts = [{"text": b["text"], "value_mm": b["value_mm"], "score": b["score"]} for b in best[1:]]
        status = "ok" if top["score"] >= 0.8 else "check"
        
        res = {
            "status": status,
            "best": top,
            "alternatives": alts,
            "process_time": elapsed_time
        }
        
    _ocr_cache[cache_key] = res
    return res

def pair_dimensions(geometry, text_group_results):
    dimensions = []
    unpaired_text_groups = []
    unlabelled_dimension_lines = []
    warnings = []
    
    vertices = {v["id"]: (v["x"], v["y"]) for v in geometry["vertices"]}
    
    dim_lines = [s for s in geometry.get("segments", []) if s["role"] == "dimension"]
    
    # Match each dim_line to closest text group
    tg_centers = {}
    for tg in geometry.get("text_groups", []):
        cx = tg["bbox"][0] + tg["bbox"][2]/2
        cy = tg["bbox"][1] + tg["bbox"][3]/2
        tg_centers[tg["id"]] = (cx, cy)
        
    dim_matches = defaultdict(list) # tg_id -> list of (dist, dim_line)
    
    for dl in dim_lines:
        p1 = vertices[dl["v1"]]
        p2 = vertices[dl["v2"]]
        cx = (p1[0] + p2[0]) / 2
        cy = (p1[1] + p2[1]) / 2
        
        best_tg = None
        best_d = float('inf')
        for tg_id, t_center in tg_centers.items():
            d = _get_distance((cx, cy), t_center)
            if d < best_d:
                best_d = d
                best_tg = tg_id
                
        if best_tg and best_d < 200: # Max pairing distance
            dim_matches[best_tg].append((best_d, dl))
        else:
            unlabelled_dimension_lines.append(dl["id"])
            warnings.append(f"Dimension line {dl['id']} has no nearby text.")
            
    # Resolve competing lines for one text group
    used_tgs = set()
    dim_count = 1
    
    for tg_id, matches in dim_matches.items():
        matches.sort(key=lambda x: x[0])
        best_dl = matches[0][1]
        dist = matches[0][0]
        
        if len(matches) > 1:
            warnings.append(f"Multiple dimension lines compete for label {tg_id}. Picked closest.")
            for _, dl in matches[1:]:
                unlabelled_dimension_lines.append(dl["id"])
                
        used_tgs.add(tg_id)
        
        # Determine measured segments
        # Find parallel outlines that match the extent
        measured_segments = []
        dl_p1 = vertices[best_dl["v1"]]
        dl_p2 = vertices[best_dl["v2"]]
        
        for s in geometry.get("segments", []):
            if s["role"] in ["outline", "feature"]:
                sp1 = vertices[s["v1"]]
                sp2 = vertices[s["v2"]]
                # If they are parallel (same orientation)
                if s["orientation"] == best_dl["orientation"] and s["orientation"] in ["H", "V"]:
                    # Project and see if extents overlap strongly
                    if s["orientation"] == "H":
                        overlap = max(0, min(max(dl_p1[0], dl_p2[0]), max(sp1[0], sp2[0])) - max(min(dl_p1[0], dl_p2[0]), min(sp1[0], sp2[0])))
                    else:
                        overlap = max(0, min(max(dl_p1[1], dl_p2[1]), max(sp1[1], sp2[1])) - max(min(dl_p1[1], dl_p2[1]), min(sp1[1], sp2[1])))
                        
                    if overlap > best_dl["length_px"] * 0.8:
                        measured_segments.append(s["id"])
                        
        res = text_group_results.get(tg_id)
        if not res:
            continue
            
        if res["status"] == "unreadable":
            b = {
                "raw_text": "",
                "normalized_text": "",
                "value_mm": 0.0,
                "unit_used": "",
                "kind": "length",
                "assumed_unit": False,
                "score": 0.0,
                "variant": "",
                "rotation": 0,
                "confidence": 0.0
            }
        else:
            b = res["best"]
            
        dimensions.append({
            "id": f"dim{dim_count}",
            "dimension_line": best_dl["id"],
            "text_group": tg_id,
            "raw_text": b["raw_text"],
            "normalized_text": b["normalized_text"],
            "value_mm": b.get("value_mm", 0.0),
            "unit_used": b.get("unit_used", ""),
            "kind": b.get("kind", "length"),
            "assumed_unit": b.get("assumed_unit", False),
            "score": b.get("score", 0.0),
            "variant": b.get("variant", ""),
            "rotation": b.get("rotation", 0),
            "confidence": b.get("confidence", 0.0),
            "status": res["status"],
            "alternatives": res.get("alternatives", []),
            "measured_segments": measured_segments,
            "pair_score": max(0.0, 1.0 - (dist / 200.0)),
            "user_edited": False,
            "process_time": res.get("process_time", 0.0)
        })
        dim_count += 1
        
    for tg in geometry.get("text_groups", []):
        if tg["id"] not in used_tgs:
            unpaired_text_groups.append(tg["id"])
            
    return dimensions, unpaired_text_groups, unlabelled_dimension_lines, warnings

def extract_dimensions(geometry, enhanced_gray, options):
    default_unit = options.get("default_unit", "mm")
    engine = options.get("engine", "easyocr")
    
    text_group_results = {}
    
    for tg in geometry.get("text_groups", []):
        crop = get_crop(enhanced_gray, tg["crop_bbox"])
        res = process_text_group(tg, crop, default_unit, engine)
        text_group_results[tg["id"]] = res
        
    dimensions, unpaired_tgs, unlabelled_dls, warnings = pair_dimensions(geometry, text_group_results)
    
    # Sheet context
    scales = []
    units_seen = set()
    for d in dimensions:
        units_seen.add(d["unit_used"])
        dl = [s for s in geometry.get("segments", []) if s["id"] == d["dimension_line"]]
        if dl and d["value_mm"] > 0:
            scales.append(dl[0]["length_px"] / d["value_mm"])
            
    scale_median = np.median(scales) if scales else 0.0
    scale_spread = 0.0
    if len(scales) > 1:
        scale_spread = (np.max(scales) - np.min(scales)) / (scale_median + 1e-5)
        if scale_spread > 0.2:
            warnings.append(f"Sketch is not to scale (spread {scale_spread:.2f}).")
            
    if len(units_seen) > 1:
        warnings.append(f"Mixed units detected on sheet: {', '.join(units_seen)}")
        
    # Check for factors of 1000 differences
    vals = [d["value_mm"] for d in dimensions if d["value_mm"] > 0]
    if vals and (max(vals) / min(vals)) > 1000:
        warnings.append("Values differ by >1000x. Check for unit confusion.")
        
    return {
        "version": 1,
        "default_unit": default_unit,
        "dimensions": dimensions,
        "unpaired_text_groups": unpaired_tgs,
        "unlabelled_dimension_lines": unlabelled_dls,
        "sheet": {
            "scale_px_per_mm_median": scale_median,
            "scale_spread": scale_spread,
            "units_seen": list(units_seen)
        },
        "warnings": warnings
    }
