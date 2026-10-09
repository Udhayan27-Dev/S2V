import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
import constraints
import solver
import model

def test_exact_l_shape():
    geometry = {
        "vertices": [
            {"id": "v1", "x": 0, "y": 0},
            {"id": "v2", "x": 240, "y": 0},
            {"id": "v3", "x": 240, "y": 100},
            {"id": "v4", "x": 120, "y": 100},
            {"id": "v5", "x": 120, "y": 200},
            {"id": "v6", "x": 0, "y": 200}
        ],
        "segments": [
            {"id": "s1", "v1": "v1", "v2": "v2", "role": "outline", "orientation": "H", "length_px": 240},
            {"id": "s2", "v1": "v2", "v2": "v3", "role": "outline", "orientation": "V", "length_px": 100},
            {"id": "s3", "v1": "v3", "v2": "v4", "role": "outline", "orientation": "H", "length_px": 120},
            {"id": "s4", "v1": "v4", "v2": "v5", "role": "outline", "orientation": "V", "length_px": 100},
            {"id": "s5", "v1": "v5", "v2": "v6", "role": "outline", "orientation": "H", "length_px": 120},
            {"id": "s6", "v1": "v6", "v2": "v1", "role": "outline", "orientation": "V", "length_px": 200}
        ]
    }
    dimensions_data = {
        "dimensions": [
            {"id": "d1", "dimension_line": "s1", "status": "ok", "value_mm": 24.0, "measured_segments": ["s1"]},
            {"id": "d2", "dimension_line": "s6", "status": "ok", "value_mm": 20.0, "measured_segments": ["s6"]},
            {"id": "d3", "dimension_line": "s3", "status": "ok", "value_mm": 12.0, "measured_segments": ["s3"]},
            {"id": "d4", "dimension_line": "s4", "status": "ok", "value_mm": 10.0, "measured_segments": ["s4"]}
        ]
    }
    
    uf_x, uf_y, inferred = constraints.build_classes(geometry)
    constrs, unbound = constraints.build_constraints(geometry, dimensions_data, uf_x, uf_y, {})
    
    cx = [c for c in constrs if c["axis"] == "x"]
    cy = [c for c in constrs if c["axis"] == "y"]
    
    solved = solver.solve(geometry, cx, cy, uf_x, uf_y, {})
    
    assert solved["dof"]["status"] == "fully_constrained"
    assert len(solved["conflicts"]) == 0
    
    m = model.build_model(geometry, solved, uf_x, uf_y, unbound, [], {"origin": "bottom_left"})
    
    v_map = {v["id"]: v for v in m["vertices"]}
    assert v_map["v1"]["x"] == 0.0
    assert v_map["v1"]["y"] == 0.0
    assert v_map["v2"]["x"] == 24.0
    assert v_map["v3"]["y"] == 10.0
    assert v_map["v5"]["y"] == 20.0

def test_conflict():
    geometry = {
        "vertices": [
            {"id": "v1", "x": 0, "y": 0},
            {"id": "v2", "x": 240, "y": 0}
        ],
        "segments": [
            {"id": "s1", "v1": "v1", "v2": "v2", "role": "outline", "orientation": "H", "length_px": 240},
            {"id": "s2", "v1": "v1", "v2": "v2", "role": "outline", "orientation": "H", "length_px": 240}
        ]
    }
    dimensions_data = {
        "dimensions": [
            {"id": "d1", "dimension_line": "s1", "status": "ok", "value_mm": 24.0, "measured_segments": ["s1"]},
            {"id": "d2", "dimension_line": "s2", "status": "ok", "value_mm": 20.0, "measured_segments": ["s2"]} # Conflict!
        ]
    }
    
    uf_x, uf_y, _ = constraints.build_classes(geometry)
    constrs, _ = constraints.build_constraints(geometry, dimensions_data, uf_x, uf_y, {})
    
    cx = [c for c in constrs if c["axis"] == "x"]
    cy = [c for c in constrs if c["axis"] == "y"]
    
    solved = solver.solve(geometry, cx, cy, uf_x, uf_y, {})
    
    assert solved["dof"]["status"] == "conflicting"
    assert len(solved["conflicts"]) > 0

def test_under_constrained():
    geometry = {
        "vertices": [
            {"id": "v1", "x": 0, "y": 0},
            {"id": "v2", "x": 100, "y": 0},
            {"id": "v3", "x": 100, "y": 100},
            {"id": "v4", "x": 0, "y": 100}
        ],
        "segments": [
            {"id": "s1", "v1": "v1", "v2": "v2", "role": "outline", "orientation": "H", "length_px": 100},
            {"id": "s2", "v1": "v2", "v2": "v3", "role": "outline", "orientation": "V", "length_px": 100},
            {"id": "s3", "v1": "v3", "v2": "v4", "role": "outline", "orientation": "H", "length_px": 100},
            {"id": "s4", "v1": "v4", "v2": "v1", "role": "outline", "orientation": "V", "length_px": 100}
        ]
    }
    # Only width is labelled
    dimensions_data = {
        "dimensions": [
            {"id": "d1", "dimension_line": "s1", "status": "ok", "value_mm": 10.0, "measured_segments": ["s1"]}
        ]
    }
    
    uf_x, uf_y, _ = constraints.build_classes(geometry)
    constrs, _ = constraints.build_constraints(geometry, dimensions_data, uf_x, uf_y, {})
    cx = [c for c in constrs if c["axis"] == "x"]
    cy = [c for c in constrs if c["axis"] == "y"]
    
    solved = solver.solve(geometry, cx, cy, uf_x, uf_y, {})
    
    assert solved["dof"]["status"] == "under_constrained"
    # Height should be estimated to 10.0 based on pixel proportions
    assert len(solved["estimates"]) > 0
    
    m = model.build_model(geometry, solved, uf_x, uf_y, [], [], {})
    v3 = next(v for v in m["vertices"] if v["id"] == "v3")
    assert v3["estimated"] == True
