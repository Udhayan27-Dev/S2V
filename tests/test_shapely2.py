from shapely.geometry import Polygon
p1 = Polygon([(599.25, 200.875), (201.375, 200.875), (200.625, 400.375), (599.75, 398.625)])
p3 = Polygon([(349.0, 250.875), (251.5, 250.875), (251.0, 349.0), (349.0, 349.0)])
print("p1 contains p3:", p1.contains(p3))
