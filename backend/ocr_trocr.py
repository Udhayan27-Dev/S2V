import os
import time
import math
import numpy as np
import cv2
import psutil
from PIL import Image

_model = None
_processor = None
_allowed_token_ids = None

# Whitelist characters
ALLOWED_CHARS = "0123456789.,/'\"’”′″- mcfintRØør"

def _load_model(size="small"):
    global _model, _processor, _allowed_token_ids
    if _model is not None:
        return _model, _processor
        
    start_time = time.time()
    
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LogitsProcessor
    
    # Configure Torch CPU
    num_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    torch.set_num_threads(num_cores)
    
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"models/trocr-{size}-handwritten")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"TrOCR model not found. Run python scripts/download_trocr.py --size {size}")
        
    _processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
    _model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
    _model.eval()
    
    # Precompute allowed token ids
    allowed_ids = set()
    # Add special tokens
    allowed_ids.add(_processor.tokenizer.pad_token_id)
    allowed_ids.add(_processor.tokenizer.bos_token_id)
    allowed_ids.add(_processor.tokenizer.eos_token_id)
    allowed_ids.add(_processor.tokenizer.unk_token_id)
    
    for i in range(_processor.tokenizer.vocab_size):
        token = _processor.tokenizer.convert_ids_to_tokens(i)
        # SentencePiece uses ' ' (U+2581) to indicate space
        token_str = token.replace(' ', '')
        
        # Check if all chars in token_str are in ALLOWED_CHARS
        if all(c in ALLOWED_CHARS for c in token_str):
            allowed_ids.add(i)
            
    _allowed_token_ids = list(allowed_ids)
    
    print(f"TrOCR {size} model loaded in {time.time() - start_time:.2f}s with {num_cores} threads")
    return _model, _processor

def unload():
    global _model, _processor, _allowed_token_ids
    _model = None
    _processor = None
    _allowed_token_ids = None

def clean_crop(crop):
    """Remove stray line fragments touching the edge."""
    h, w = crop.shape
    # Binarize to find dark pixels (strokes)
    _, bin_crop = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_crop, connectivity=8)
    
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        touches_edge = (x == 0 or y == 0 or x + bw == w or y + bh == h)
        is_long = max(bw, bh) > min(w, h) * 0.5
        is_thin = min(bw, bh) < 5
        
        if touches_edge and (is_long or is_thin):
            continue # drop it
        
        mask[labels == i] = crop[labels == i]
        
    return mask

def make_letterbox(crop_rgb, target_size=384):
    h, w = crop_rgb.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    
    square = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    square[y_off:y_off+nh, x_off:x_off+nw] = resized
    return square

class WhitelistLogitsProcessor:
    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids

    def __call__(self, input_ids, scores):
        import torch
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.allowed_ids] = False
        scores[mask] = -float("inf")
        return scores

def recognize_candidates(crop_gray, orientation="H", model_size="small", use_whitelist=True, use_letterbox=True):
    import torch
    from transformers import LogitsProcessorList
    
    model, processor = _load_model(model_size)
    
    # 1. Clean
    cleaned = clean_crop(crop_gray)
    
    # 2. RGB and Pad
    rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]
    pad_h = int(h * 0.15)
    pad_w = int(w * 0.15)
    padded = cv2.copyMakeBorder(rgb, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255,255,255])
    
    # 3. Candidates
    candidates_imgs = []
    
    def add_rotations(img, variant_name):
        candidates_imgs.append((img, 0, variant_name))
        candidates_imgs.append((cv2.rotate(img, cv2.ROTATE_180), 180, variant_name))
        if orientation == "V":
            candidates_imgs.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90, variant_name))
            candidates_imgs.append((cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 270, variant_name))

    add_rotations(padded, "tight")
    if use_letterbox:
        add_rotations(make_letterbox(padded), "letterbox")
        
    def sort_key(c):
        rot = c[1]
        var = c[2]
        score = 0
        if rot == 0: score -= 10
        if var == "letterbox": score -= 5
        return score
        
    candidates_imgs.sort(key=sort_key)
    candidates_imgs = candidates_imgs[:6]
    
    logits_processor = LogitsProcessorList()
    if use_whitelist:
        logits_processor.append(WhitelistLogitsProcessor(_allowed_token_ids))
        
    with torch.inference_mode():
        for img, rot, var in candidates_imgs:
            pil_img = Image.fromarray(img)
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
            
            outputs = model.generate(
                pixel_values,
                num_beams=4,
                max_new_tokens=16,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
                num_return_sequences=3
            )
            
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
            )
            
            results_for_cand = []
            for i in range(len(outputs.sequences)):
                seq = outputs.sequences[i]
                scores = transition_scores[i]
                
                valid_mask = (seq != processor.tokenizer.pad_token_id) & \
                             (seq != processor.tokenizer.bos_token_id) & \
                             (seq != processor.tokenizer.eos_token_id)
                gen_seq = seq[1:]
                valid_mask = valid_mask[1:]
                
                if not valid_mask.any():
                    continue
                    
                valid_scores = scores[valid_mask]
                mean_log_prob = valid_scores.mean().item()
                min_log_prob = valid_scores.min().item()
                
                confidence = math.exp(mean_log_prob)
                min_prob = math.exp(min_log_prob)
                
                text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
                
                results_for_cand.append({
                    "text": text,
                    "confidence": confidence,
                    "min_token_prob": min_prob,
                    "rotation": rot,
                    "variant": var,
                    "beam_index": i
                })
            
            yield results_for_cand
