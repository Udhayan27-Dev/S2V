from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize

lines = [
    LineString([(0, 0), (10, 0)]),
    LineString([(10, 0), (10, 10)]),
    LineString([(10, 10), (0, 10)]),
    LineString([(0, 10), (0, 0)]),
    LineString([(3, 3), (7, 3)]),
    LineString([(7, 3), (7, 7)]),
    LineString([(7, 7), (3, 7)]),
    LineString([(3, 7), (3, 3)])
]
merged = unary_union(lines)
polys = list(polygonize(merged))
for i, p in enumerate(polys):
    print(f"Poly {i} area: {p.area}")
    
print("p0 contains p1:", polys[0].contains(polys[1]))
print("p1 contains p0:", polys[1].contains(polys[0]))
