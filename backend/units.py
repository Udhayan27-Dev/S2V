import re

def to_mm(text, default_unit="mm"):
    try:
        return _to_mm(text, default_unit)
    except ValueError:
        return None

def _to_mm(text, default_unit="mm"):
    original_text = text.strip()
    if not original_text:
        return None
        
    text = original_text.lower()
    
    kind = "length"
    # Check prefixes
    if text.startswith("ø") or text.startswith("⌀"):
        kind = "diameter"
        text = text[1:].strip()
    elif text.startswith("r") and (len(text) == 1 or not text[1].isalpha()):
        # Ensure it's R followed by number or space
        kind = "radius"
        text = text[1:].strip()

    # Replace fancy quotes
    text = text.replace("’", "'").replace("”", '"')
    
    # Replace comma with dot for decimals (e.g. 12,5 -> 12.5)
    # But be careful not to break "1,000" if we expect thousands. 
    # Usually in these sketches, comma is a decimal separator.
    text = re.sub(r'(\d),(\d)', r'\1.\2', text)
    
    # Parse metric
    m_match = re.match(r'^([0-9\.]+)\s*(mm|cm|m)$', text)
    if m_match:
        val = float(m_match.group(1))
        unit = m_match.group(2)
        if unit == "cm":
            val *= 10
        elif unit == "m":
            val *= 1000
        return {
            "value_mm": val,
            "unit_used": unit,
            "kind": kind,
            "assumed": False,
            "normalized_text": f"{val:g} {unit}"
        }
        
    # Parse bare number
    bare_match = re.match(r'^([0-9\.]+)$', text)
    if bare_match:
        val = float(bare_match.group(1))
        unit = default_unit
        if unit == "cm":
            val_mm = val * 10
        elif unit == "m":
            val_mm = val * 1000
        elif unit == "in":
            val_mm = val * 25.4
        elif unit == "ft":
            val_mm = val * 304.8
        else: # mm
            val_mm = val
            unit = "mm"
        return {
            "value_mm": val_mm,
            "unit_used": unit,
            "kind": kind,
            "assumed": True,
            "normalized_text": f"{val:g} {unit}"
        }
        
    # Parse fractional inches (e.g., 2 1/2", 2 1/2 in)
    frac_match = re.match(r'^([0-9]+)\s+([0-9]+)\s*/\s*([0-9]+)\s*(?:\"|in)$', text)
    if frac_match:
        whole = float(frac_match.group(1))
        num = float(frac_match.group(2))
        den = float(frac_match.group(3))
        val_in = whole + num / den
        return {
            "value_mm": val_in * 25.4,
            "unit_used": "in",
            "kind": kind,
            "assumed": False,
            "normalized_text": f"{val_in:g} in"
        }
        
    # Parse bare inches (e.g., 6", 6 in, 5.5")
    in_match = re.match(r'^([0-9\.]+)\s*(?:\"|in)$', text)
    if in_match:
        val = float(in_match.group(1))
        return {
            "value_mm": val * 25.4,
            "unit_used": "in",
            "kind": kind,
            "assumed": False,
            "normalized_text": f"{val:g} in"
        }
        
    # Parse bare feet (e.g., 5.5', 5 ft)
    ft_match = re.match(r'^([0-9\.]+)\s*(?:\'|ft)$', text)
    if ft_match:
        val = float(ft_match.group(1))
        return {
            "value_mm": val * 304.8,
            "unit_used": "ft",
            "kind": kind,
            "assumed": False,
            "normalized_text": f"{val:g} ft"
        }
        
    # Parse feet and inches (e.g., 5'6", 5'-6", 5 ft 6 in)
    ft_in_match = re.match(r'^([0-9\.]+)\s*(?:\'|ft)\s*-?\s*([0-9\.]+)\s*(?:\"|in)$', text)
    if ft_in_match:
        ft = float(ft_in_match.group(1))
        inch = float(ft_in_match.group(2))
        val_in = ft * 12 + inch
        return {
            "value_mm": val_in * 25.4,
            "unit_used": "ft-in",
            "kind": kind,
            "assumed": False,
            "normalized_text": f"{ft:g} ft {inch:g} in"
        }
        
    return None
