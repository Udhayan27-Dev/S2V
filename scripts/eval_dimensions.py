import os
import json
import argparse
import sys

# Ensure backend imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

import cv2
from enhance import enhance
import geometry
import dimensions

def evaluate(images_dir, gt_path, engine="easyocr"):
    if not os.path.exists(gt_path):
        print(f"Ground truth file not found: {gt_path}")
        return
        
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
        
    total_expected = 0
    total_found = 0
    correct_matches = 0
    
    for filename, expected_dims in ground_truth.items():
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\nProcessing {filename}...")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("Failed to read image")
            continue
            
        # 1. Enhance
        enh_gray, binary = enhance(img_bgr)
        
        # 2. Geometry
        params = {
            "min_segment_len": max(img_bgr.shape) * 0.012,
            "spur_prune": max(img_bgr.shape) * 0.01,
            "snap_angle_deg": 8.0,
            "merge_dist": max(img_bgr.shape) * 0.012,
            "text_max_size": min(img_bgr.shape) * 0.06
        }
        geom = geometry.extract_geometry(binary, enh_gray, params)
        
        # 3. Dimensions
        options = {"engine": engine, "default_unit": "mm"}
        dim_data = dimensions.extract_dimensions(geom, enh_gray, options)
        
        found_vals = [d["value_mm"] for d in dim_data["dimensions"] if d["status"] != "unreadable"]
        
        print(f"  Expected: {expected_dims}")
        print(f"  Found: {found_vals}")
        
        total_expected += len(expected_dims)
        total_found += len(found_vals)
        
        # Simple greedy matching
        matched_expected = set()
        for fv in found_vals:
            best_match = None
            best_diff = float('inf')
            
            for i, ev in enumerate(expected_dims):
                if i not in matched_expected:
                    diff = abs(fv - ev)
                    if diff < best_diff and diff < 1.0: # allow small float diff
                        best_diff = diff
                        best_match = i
                        
            if best_match is not None:
                matched_expected.add(best_match)
                correct_matches += 1
                
    print("\n--- RESULTS ---")
    precision = correct_matches / total_found if total_found > 0 else 0
    recall = correct_matches / total_expected if total_expected > 0 else 0
    print(f"Precision: {precision:.2f} ({correct_matches}/{total_found} predicted correct)")
    print(f"Recall:    {recall:.2f} ({correct_matches}/{total_expected} ground truths found)")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dimension OCR")
    parser.add_argument("--dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground_truth.json")
    parser.add_argument("--engine", type=str, default="easyocr", choices=["easyocr", "tesseract", "both"])
    
    args = parser.parse_args()
    evaluate(args.dir, args.gt, args.engine)
