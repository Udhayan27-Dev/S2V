import pytest
from backend.dimensions import clean_text, pair_dimensions

def test_clean_text():
    # Only within digit contexts or small tokens
    c, corr = clean_text("l2 rnm")
    assert c == "12 mm"
    assert corr is True
    
    c, corr = clean_text("1O mm")
    assert c == "10 mm"
    assert corr is True
    
    c, corr = clean_text("S mm")
    assert c == "5 mm"
    assert corr is True
    
    c, corr = clean_text("2Z Bq")
    assert c == "22 89"
    
    c, corr = clean_text("Not a number")
    assert c == "Not a number"
    assert corr is False
    
def test_pair_dimensions():
    # Minimal mock geometry
    geometry = {
        "vertices": [
            {"id": "v1", "x": 0, "y": 0},
            {"id": "v2", "x": 100, "y": 0},
            {"id": "v3", "x": 0, "y": 10},
            {"id": "v4", "x": 100, "y": 10}
        ],
        "segments": [
            {"id": "d1", "v1": "v1", "v2": "v2", "role": "dimension", "length_px": 100, "orientation": "H"},
            {"id": "o1", "v1": "v3", "v2": "v4", "role": "outline", "length_px": 100, "orientation": "H"}
        ],
        "text_groups": [
            {"id": "t1", "bbox": [40, -10, 20, 10]}
        ]
    }
    
    tg_results = {
        "t1": {
            "status": "ok",
            "best": {
                "raw_text": "100 mm",
                "normalized_text": "100 mm",
                "value_mm": 100.0,
                "unit_used": "mm",
                "kind": "length",
                "assumed_unit": False,
                "score": 0.9
            },
            "alternatives": []
        }
    }
    
    dimensions, unpaired, unlabelled, warnings = pair_dimensions(geometry, tg_results)
    
    assert len(dimensions) == 1
    d = dimensions[0]
    assert d["dimension_line"] == "d1"
    assert d["text_group"] == "t1"
    assert "o1" in d["measured_segments"]
    assert d["value_mm"] == 100.0
    assert len(unpaired) == 0
    assert len(unlabelled) == 0
