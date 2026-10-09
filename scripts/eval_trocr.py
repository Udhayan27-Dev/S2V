import os
import sys
import time
import cv2
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from page import detect_page, straighten
from enhance import enhance
from geometry import extract_geometry
import ocr_trocr
from units import to_mm

TRUE_VALUES = [12.0, 20.0, 10.0, 24.0, 24.0, 5.0, 10.0]

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_eval(img_path, model_size, use_whitelist, use_letterbox):
    print(f"\n--- EVALUATING TrOCR {model_size.upper()} | Whitelist: {use_whitelist} | Letterbox: {use_letterbox} ---")
    
    img_bgr = cv2.imread(img_path)
    corners, _, _, _ = detect_page(img_bgr)
    straightened = straighten(img_bgr, corners)
    bin_img, enhanced_gray = enhance(straightened)
    geometry = extract_geometry(bin_img, enhanced_gray, {})
    
    # Filter noise text groups to only evaluate likely labels
    all_text_groups = geometry.get("text_groups", [])
    text_groups = [tg for tg in all_text_groups if tg["bbox"][2] >= 15 and tg["bbox"][3] >= 15]
    
    print(f"Extracted {len(all_text_groups)} text crops, evaluating {len(text_groups)}.")
    
    # Load model (lazy load) to not affect crop timing, and measure peak memory
    mem_before = get_memory_mb()
    load_start = time.time()
    ocr_trocr._load_model(model_size)
    load_end = time.time()
    mem_after = get_memory_mb()
    
    print(f"Model Load Time: {load_end - load_start:.2f}s")
    print(f"Memory Increase: {mem_after - mem_before:.2f} MB")
    
    if len(text_groups) != len(TRUE_VALUES):
        print(f"WARNING: Expected {len(TRUE_VALUES)} crops, got {len(text_groups)}.")
        
    correct = 0
    total = len(text_groups)
    times = []
    
    failed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../failed_crops'))
    os.makedirs(failed_dir, exist_ok=True)
    
    for i, tg in enumerate(text_groups):
        bbox = tg["bbox"]
        x, y, w, h = bbox
        crop_gray = enhanced_gray[y:y+h, x:x+w]
        ori = tg.get("orientation", "H")
        
        start_t = time.time()
        
        # We simulate the exact logic in dimensions.py loop
        generator = ocr_trocr.recognize_candidates(crop_gray, ori, model_size, use_whitelist, use_letterbox)
        best_val = None
        best_conf = 0
        best_text = ""
        
        for batch in generator:
            if isinstance(batch, dict): batch = [batch]
            batch_scored = []
            for c in batch:
                txt = c["text"]
                conf = c["confidence"]
                parsed = to_mm(txt, "mm")
                if parsed:
                    batch_scored.append({"val": parsed["value_mm"], "conf": conf, "text": txt})
            
            if batch_scored:
                b = max(batch_scored, key=lambda x: x["conf"])
                if b["conf"] > best_conf:
                    best_conf = b["conf"]
                    best_val = b["val"]
                    best_text = b["text"]
                
                # early stopping threshold
                if best_conf >= 0.8:
                    break
                    
        elapsed = time.time() - start_t
        times.append(elapsed)
        
        # Check against true values
        true_val = TRUE_VALUES[i] if i < len(TRUE_VALUES) else None
        
        is_correct = (best_val == true_val)
        if is_correct:
            correct += 1
            print(f"Crop {i}: CORRECT ({best_val} mm) in {elapsed:.2f}s")
        else:
            print(f"Crop {i}: FAILED (Expected {true_val}, Got {best_val} '{best_text}') in {elapsed:.2f}s")
            # Save failed crop
            fail_path = os.path.join(failed_dir, f"fail_{model_size}_wl{use_whitelist}_lb{use_letterbox}_crop{i}.png")
            cv2.imwrite(fail_path, crop_gray)
            
    if total > 0:
        print(f"Accuracy: {correct}/{total} ({(correct/total)*100:.1f}%)")
        print(f"Avg Time per crop: {sum(times)/len(times):.2f}s")
    else:
        print("Accuracy: N/A (0 crops)")
    print(f"Total Time: {sum(times):.2f}s")
    
    # unload to free memory for next run
    ocr_trocr.unload()

def main():
    img_path = '/home/udhayan/.gemini/antigravity/brain/d1b1b3d8-4505-4a9c-9123-8d2e28cf7181/media__1791487731375.png'
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
    print(f"Running eval on: {img_path}")
    
    # Grid search
    for size in ["small"]:
        for lb in [True, False]:
            for wl in [True, False]:
                run_eval(img_path, size, use_whitelist=wl, use_letterbox=lb)
                
if __name__ == '__main__':
    main()
