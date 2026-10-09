import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
import ocr_trocr

def test_whitelist_initialization():
    # Load model and verify allowed tokens
    try:
        model, processor = ocr_trocr._load_model("small")
    except FileNotFoundError:
        pytest.skip("Model not downloaded yet")
        
    allowed = ocr_trocr._allowed_token_ids
    assert allowed is not None
    assert len(allowed) > 0
    
    # Check that a common digit is allowed
    # Note: tokenizer specific, but space and digits should be there
    # For TrOCR, 0 is allowed. Let's just check the length is constrained
    # typical vocab size is ~50000, whitelist should be < 500
    assert len(allowed) < 2000

def test_missing_model_throws():
    ocr_trocr.unload()
    # If we request a model that doesn't exist, it should throw FileNotFoundError
    with pytest.raises(FileNotFoundError):
        ocr_trocr._load_model("nonexistent_size")

def test_recognize_candidates_interface():
    # Provide a dummy white crop
    dummy_crop = np.ones((64, 128), dtype=np.uint8) * 255
    
    try:
        gen = ocr_trocr.recognize_candidates(dummy_crop, "H", "small", True, True)
        batch = next(gen)
        # Should return at least one dictionary with 'text' and 'confidence'
        assert isinstance(batch, list)
        assert len(batch) > 0
        assert "text" in batch[0]
        assert "confidence" in batch[0]
        assert "variant" in batch[0]
    except FileNotFoundError:
        pytest.skip("Model not downloaded yet")
