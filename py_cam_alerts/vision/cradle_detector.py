import cv2
import numpy as np

class CradleDetector:
    """
    Monitors toddler position inside or near a defined Cradle / Bed ROI boundary.
    Detects boundary approach, climbing out attempt, and exit/fall danger.
    """
    def __init__(self, roi_polygon=None):
        # Default ROI ratio: centered crib region (x: 20-80%, y: 30-85% of frame)
        self.roi_polygon = roi_polygon
        self.default_normalized_roi = [[0.2, 0.3], [0.8, 0.3], [0.8, 0.85], [0.2, 0.85]]

    def get_roi_pixels(self, frame_w, frame_h):
        if self.roi_polygon is not None and len(self.roi_polygon) >= 3:
            return np.array(self.roi_polygon, dtype=np.int32)
        
        # Use normalized default scaled to current frame size
        pts = [[int(pt[0] * frame_w), int(pt[1] * frame_h)] for pt in self.default_normalized_roi]
        return np.array(pts, dtype=np.int32)

    def set_custom_roi(self, polygon_pts):
        """polygon_pts: list of [x, y] coordinates in pixels"""
        if polygon_pts and len(polygon_pts) >= 3:
            self.roi_polygon = polygon_pts

    def analyze_child(self, bbox, frame_w, frame_h, keypoints=None):
        """
        Analyze child bounding box and keypoints relative to cradle/bed ROI.
        returns dict with:
           status (str): "SAFE_INSIDE" | "APPROACHING_EDGE" | "CLIMBING_DANGER" | "OUT_OF_BED"
           is_danger (bool)
           distance_to_boundary (float)
           details (dict)
        """
        x1, y1, x2, y2 = bbox
        child_cx = (x1 + x2) / 2.0
        child_cy = (y1 + y2) / 2.0
        head_y = y1

        roi_pts = self.get_roi_pixels(frame_w, frame_h)
        
        # Test if centroid is inside polygon
        dist = cv2.pointPolygonTest(roi_pts, (child_cx, child_cy), measureDist=True)
        
        # Test if head is inside polygon
        head_dist = cv2.pointPolygonTest(roi_pts, (child_cx, head_y), measureDist=True)

        is_danger = False
        status = "SAFE_INSIDE"
        alert_msg = ""

        # dist > 0 means inside polygon; dist < 0 means outside polygon; dist == 0 on edge
        if dist < -20:
            # Fully outside cradle boundary
            status = "OUT_OF_BED"
            is_danger = True
            alert_msg = "ALERT: Toddler is OUT OF BED / CRADLE!"
        elif dist < 30 or head_dist < -10:
            # Head or torso reaching/crossing boundary top
            status = "CLIMBING_DANGER"
            is_danger = True
            alert_msg = "WARNING: Toddler climbing cradle wall / exit risk!"
        elif dist < 60:
            # Approaching boundary edge
            status = "APPROACHING_EDGE"
            is_danger = False
            alert_msg = "Toddler moving near cradle edge"
        else:
            status = "SAFE_INSIDE"
            is_danger = False
            alert_msg = "Toddler safe inside cradle"

        return {
            "status": status,
            "is_danger": is_danger,
            "dist_to_boundary": round(dist, 1),
            "alert_msg": alert_msg,
            "roi_points": roi_pts.tolist()
        }
