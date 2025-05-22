import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in [top-left, top-right, bottom-right, bottom-left] order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def unclip_polygon(poly: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """Expand polygon by unclip_ratio."""
    area = Polygon(poly).area
    length = Polygon(poly).length
    if length == 0 or area == 0:
        return poly
    offset = (area * unclip_ratio) / length
    pc = pyclipper.PyclipperOffset()
    pc.AddPath(poly.astype(np.int32).tolist(),
               pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pc.Execute(offset)
    if not expanded:
        return poly
    return np.array(max(expanded, key=lambda p: Polygon(p).area),
                    dtype=np.float32)

def warp_crop(img: np.ndarray, box: np.ndarray, height: int) -> np.ndarray:
    """Perspective-warp quadrilateral to rectangle of given height."""
    box = order_points(box)
    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    w = int(max(widthA, widthB))
    h = height
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)