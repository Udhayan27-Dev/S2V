import ocr_trocr

def recognize(crop_gray, orientation="H", engine="trocr-small", use_whitelist=True, use_letterbox=True):
    """
    Takes the grayscale crop of the text group, processes it, and yields candidates.
    engine: "trocr-small" or "trocr-base"
    Returns an iterator of candidates: {text, confidence, min_token_prob, rotation, variant}
    """
    model_size = "base" if engine == "trocr-base" else "small"
    
    # We yield candidate batches from ocr_trocr
    for batch in ocr_trocr.recognize_candidates(
        crop_gray, 
        orientation, 
        model_size=model_size, 
        use_whitelist=use_whitelist, 
        use_letterbox=use_letterbox
    ):
        for cand in batch:
            yield cand
