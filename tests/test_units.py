import pytest
from backend.units import to_mm

def test_to_mm_metric():
    assert to_mm("12 mm")["value_mm"] == 12.0
    assert to_mm("12mm")["value_mm"] == 12.0
    assert to_mm("30 cm")["value_mm"] == 300.0
    assert to_mm("1.2 m")["value_mm"] == 1200.0
    assert to_mm("12,5 mm")["value_mm"] == 12.5
    
def test_to_mm_imperial():
    # 5'6" = 66 inches = 1676.4 mm
    assert to_mm("5'6\"")["value_mm"] == pytest.approx(1676.4)
    assert to_mm("5’6”")["value_mm"] == pytest.approx(1676.4)
    assert to_mm("5'-6\"")["value_mm"] == pytest.approx(1676.4)
    assert to_mm("5 ft 6 in")["value_mm"] == pytest.approx(1676.4)
    assert to_mm("5.5'")["value_mm"] == pytest.approx(1676.4)
    
    # Inches
    assert to_mm("6\"")["value_mm"] == pytest.approx(152.4)
    assert to_mm("2 1/2\"")["value_mm"] == pytest.approx(63.5)

def test_to_mm_assumed():
    res = to_mm("24")
    assert res["value_mm"] == 24.0
    assert res["assumed"] is True
    
    # With a different default
    res = to_mm("24", default_unit="cm")
    assert res["value_mm"] == 240.0
    assert res["assumed"] is True
    
def test_to_mm_prefixes():
    res = to_mm("Ø10 mm")
    assert res["value_mm"] == 10.0
    assert res["kind"] == "diameter"
    
    res = to_mm("R5")
    assert res["value_mm"] == 5.0
    assert res["kind"] == "radius"
    assert res["assumed"] is True

def test_to_mm_invalid():
    assert to_mm("abc") is None
