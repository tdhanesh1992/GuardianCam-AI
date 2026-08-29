import cv2
import numpy as np

class HazardDetector:
    """
    Detects environmental hazards surrounding children:
    1. Liquid/Water/Oil spills using HSV specular & reflection segmentation.
    2. Proximity to dangerous or sharp object bounding boxes.
    """
    def __init__(self):
        # COCO object classes that pose potential danger to toddlers/children
        self.dangerous_classes = {
            "knife", "scissors", "bottle", "cup", "laptop",
            "tv", "microwave", "oven", "toaster", "sink"
        }

    def detect_spills(self, frame):
        """
        Segment wet / liquid / water / oil spill zones on floor surface.
        Uses HSV specular reflection, luminance contrast, and thresholding.
        Returns list of spill bounding boxes [[x1, y1, x2, y2, area, type]]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Liquid spill characteristic 1: High value (specular reflection) with low saturation (water gleam)
        # Liquid spill characteristic 2: Smooth specular variance surrounded by textured background
        _, high_v = cv2.threshold(v, 215, 255, cv2.THRESH_BINARY)
        _, low_s = cv2.threshold(s, 65, 255, cv2.THRESH_BINARY_INV)
        
        spill_mask = cv2.bitwise_and(high_v, low_s)

        # Apply morphological operations to connect liquid droplets/blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        spill_mask = cv2.morphologyEx(spill_mask, cv2.MORPH_CLOSE, kernel)
        spill_mask = cv2.morphologyEx(spill_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(spill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spills = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and huge background areas
            if 400 < area < (frame_area * 0.15):
                x, y, w, h_box = cv2.boundingRect(cnt)
                spills.append({
                    "bbox": [x, y, x + w, y + h_box],
                    "area": int(area),
                    "type": "Water/Liquid Spill"
                })

        return spills, spill_mask

    def check_hazards_and_proximity(self, frame, child_bboxes, detected_objects=None):
        """
        Analyze child bounding boxes against spills and hazardous objects.
        returns (hazards_found, alerts_list)
        """
        spills, spill_mask = self.detect_spills(frame)
        alerts = []
        all_hazards = []

        # Process Spills
        for s in spills:
            sx1, sy1, sx2, sy2 = s["bbox"]
            all_hazards.append({
                "type": "LIQUID_SPILL",
                "label": "Liquid/Oil Spill",
                "bbox": s["bbox"],
                "risk_level": "WARNING"
            })

            # Check proximity to any child
            for cb in child_bboxes:
                cx1, cy1, cx2, cy2 = cb
                child_feet_x = (cx1 + cx2) / 2.0
                child_feet_y = cy2  # Bottom of bounding box = feet

                spill_cx = (sx1 + sx2) / 2.0
                spill_cy = (sy1 + sy2) / 2.0

                dist = np.hypot(child_feet_x - spill_cx, child_feet_y - spill_cy)
                if dist < 120:  # within 120 pixels proximity
                    alerts.append({
                        "hazard_type": "SLIPPERY_SPILL",
                        "title": "Hazard Warning: Slippery Surface",
                        "description": "Child near water/oil spill! High slipping risk.",
                        "severity": "HIGH",
                        "bbox": s["bbox"]
                    })

        # Process Detected Object Hazards (if any from object detector)
        if detected_objects:
            for obj in detected_objects:
                label = obj.get("label", "").lower()
                obbox = obj.get("bbox", [])
                if label in self.dangerous_classes and len(obbox) == 4:
                    all_hazards.append({
                        "type": "DANGEROUS_OBJECT",
                        "label": f"Hazard: {label.capitalize()}",
                        "bbox": obbox,
                        "risk_level": "WARNING"
                    })

                    # Check proximity to children
                    ox1, oy1, ox2, oy2 = obbox
                    obj_cx = (ox1 + ox2) / 2.0
                    obj_cy = (oy1 + oy2) / 2.0

                    for cb in child_bboxes:
                        cx1, cy1, cx2, cy2 = cb
                        child_cx = (cx1 + cx2) / 2.0
                        child_cy = (cy1 + cy2) / 2.0
                        dist = np.hypot(child_cx - obj_cx, child_cy - obj_cy)
                        if dist < 100:
                            alerts.append({
                                "hazard_type": "DANGEROUS_OBJECT",
                                "title": f"Hazard Warning: {label.capitalize()} Proximity",
                                "description": f"Child close to potentially dangerous item: {label}",
                                "severity": "MEDIUM",
                                "bbox": obbox
                            })

        # Track new alerts vs already active alerts to prevent stat spam
        if not hasattr(self, "active_alert_keys"):
            self.active_alert_keys = set()

        current_keys = set()
        new_alert_count = 0
        for a in alerts:
            k = f"{a['hazard_type']}_{int(a['bbox'][0] // 40)}_{int(a['bbox'][1] // 40)}"
            current_keys.add(k)
            if k not in self.active_alert_keys:
                new_alert_count += 1
        self.active_alert_keys = current_keys

        return all_hazards, alerts, spill_mask, new_alert_count
