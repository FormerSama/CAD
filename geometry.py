from typing import List, Tuple

class PolygonRegion:
    id: int                          # 第幾個區塊
    area: float                      # 面積 (像素數)
    centroid: Tuple[float, float]    # (cx, cy)
    polygon: List[Tuple[int, int]]   # 多邊形頂點座標列表 [(x1,y1), (x2,y2), ...]

    def __init__(self, id: int, area: float, centroid: Tuple[float, float], polygon: List[Tuple[int, int]]):
        self.id = id
        self.area = area
        self.centroid = centroid
        self.polygon = polygon
        
