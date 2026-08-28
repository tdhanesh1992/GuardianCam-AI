import cv2
import time
import numpy as np
import logging

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from .fall_detector import FallDetector
from .cradle_detector import CradleDetector
from .hazard_detector import HazardDetector

logger = logging.getLogger("ChildMonitoringEngine")

class ChildMonitoringEngine:
    def __init__(self, mode="play_area"):
        """
        mode: "cradle" | "play_area" | "all"
        """
        self.mode = mode
        self.fall_detector = FallDetector()
        self.cradle_detector = CradleDetector()
        self.hazard_detector = HazardDetector()
        
        self.yolo_pose = None
        self.yolo_obj = None
        self._init_models()

        # Engine Stats & State
        self.stats = {
            "fps": 0.0,
            "children_count": 0,
            "fall_events": 0,
            "cradle_breaches": 0,
            "hazard_alerts": 0,
            "overall_status": "SAFE",  # SAFE, WARNING, DANGER
            "last_alert": None
        }

        # Overlay control toggles
        self.draw_boxes = True
        self.draw_skeletons = True
        self.draw_cradle_roi = True
        self.draw_spills = True

    def _init_models(self):
        if HAS_YOLO:
            try:
                # Load lightweight nano models
                self.yolo_pose = YOLO("yolov8n-pose.pt")
                self.yolo_obj = YOLO("yolov8n.pt")
                logger.info("YOLOv8 pose & object detection models loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load official YOLO weights ({e}). Operating in open-source OpenCV detection fallback mode.")
                self.yolo_pose = None
                self.yolo_obj = None

    def process_frame(self, frame):
        """
        Process a single BGR frame image.
        Returns:
          annotated_frame (BGR np.ndarray), frame_analysis (dict)
        """
        start_time = time.time()
        frame_h, frame_w = frame.shape[:2]

        annotated = frame.copy()
        children_detected = []
        child_bboxes = []
        detected_objects = []
        active_alerts = []

        # 1. AI Inference (YOLO or OpenCV Fallback)
        if self.yolo_pose is not None:
            # Run YOLO Pose for person tracking
            results = self.yolo_pose(frame, verbose=False, conf=0.35)
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                keypoints_data = None
                if results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints.xy.cpu().numpy()

                for idx, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    if int(cls) == 0:  # Class 0 = Person
                        kpts = keypoints_data[idx] if (keypoints_data is not None and idx < len(keypoints_data)) else None
                        child_bboxes.append(box.astype(int).tolist())
                        children_detected.append({
                            "id": idx + 1,
                            "bbox": box.astype(int).tolist(),
                            "keypoints": kpts,
                            "conf": float(conf)
                        })

            # Run YOLO Object Detection for hazard objects
            obj_results = self.yolo_obj(frame, verbose=False, conf=0.3)
            if len(obj_results) > 0 and obj_results[0].boxes is not None:
                oboxes = obj_results[0].boxes.xyxy.cpu().numpy()
                oclasses = obj_results[0].boxes.cls.cpu().numpy()
                names = obj_results[0].names
                for obox, ocls in zip(oboxes, oclasses):
                    name = names.get(int(ocls), "")
                    detected_objects.append({
                        "label": name,
                        "bbox": obox.astype(int).tolist()
                    })

        else:
            # OpenCV Fallback Person Detection using HOG / Color / Motion analysis
            children_detected, child_bboxes = self._opencv_fallback_person_detection(frame)

        # 2. Scenario 1 Check: Cradle / Bed Exit Monitoring
        cradle_info = None
        if self.mode in ("cradle", "all") or len(child_bboxes) > 0:
            for child in children_detected:
                c_analysis = self.cradle_detector.analyze_child(
                    child["bbox"], frame_w, frame_h, child.get("keypoints")
                )
                child["cradle_analysis"] = c_analysis
                cradle_info = c_analysis
                if c_analysis["is_danger"]:
                    self.stats["cradle_breaches"] += 1
                    active_alerts.append({
                        "id": f"cradle_{int(time.time()*1000)}",
                        "category": "CRADLE_BREACH",
                        "title": c_analysis["alert_msg"],
                        "severity": "CRITICAL" if c_analysis["status"] == "OUT_OF_BED" else "HIGH",
                        "timestamp": time.strftime("%H:%M:%S")
                    })

        # 3. Scenario 2 Check: Play Area Fall Detection
        for child in children_detected:
            fall_res = self.fall_detector.analyze_pose(
                child["id"], child["bbox"], child.get("keypoints")
            )
            child["fall_analysis"] = fall_res
            if fall_res["is_fall"]:
                self.stats["fall_events"] += 1
                active_alerts.append({
                    "id": f"fall_{child['id']}_{int(time.time()*1000)}",
                    "category": "FALL_DETECTED",
                    "title": f"ALERT: Child #{child['id']} has Fallen Down!",
                    "severity": "CRITICAL",
                    "timestamp": time.strftime("%H:%M:%S")
                })

        # 4. Scenario 3 Check: Surroundings & Liquid Spill Hazards
        hazards, hazard_alerts, spill_mask = self.hazard_detector.check_hazards_and_proximity(
            frame, child_bboxes, detected_objects
        )
        if len(hazard_alerts) > 0:
            self.stats["hazard_alerts"] += len(hazard_alerts)
            for ha in hazard_alerts:
                active_alerts.append({
                    "id": f"hazard_{int(time.time()*1000)}",
                    "category": "HAZARD_PROXIMITY",
                    "title": ha["title"],
                    "severity": ha["severity"],
                    "timestamp": time.strftime("%H:%M:%S")
                })

        # 5. Determine Overall Safety Status
        if any(a["severity"] == "CRITICAL" for a in active_alerts):
            overall = "DANGER"
        elif len(active_alerts) > 0:
            overall = "WARNING"
        else:
            overall = "SAFE"

        self.stats["overall_status"] = overall
        self.stats["children_count"] = len(children_detected)
        if len(active_alerts) > 0:
            self.stats["last_alert"] = active_alerts[0]

        # 6. Render Overlays on Frame
        annotated = self._draw_overlays(
            annotated, children_detected, hazards, spill_mask, active_alerts, overall
        )

        elapsed = time.time() - start_time
        self.stats["fps"] = round(1.0 / max(0.001, elapsed), 1)

        return annotated, {
            "stats": self.stats,
            "children": children_detected,
            "cradle": cradle_info,
            "hazards": hazards,
            "alerts": active_alerts
        }

    def _opencv_fallback_person_detection(self, frame):
        """Fallback detector when PyTorch models are unavailable."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Adaptive thresholding to detect moving foreground shapes
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        children = []
        bboxes = []
        c_id = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (h * w * 0.02) < area < (h * w * 0.4):
                x, y, bw, bh = cv2.boundingRect(cnt)
                bbox = [x, y, x + bw, y + bh]
                bboxes.append(bbox)
                children.append({
                    "id": c_id,
                    "bbox": bbox,
                    "keypoints": None,
                    "conf": 0.75
                })
                c_id += 1
                if c_id > 4:
                    break
        return children, bboxes

    def _draw_overlays(self, frame, children, hazards, spill_mask, alerts, overall):
        h, w = frame.shape[:2]

        # Draw Liquid Spill Highlight Mask
        if self.draw_spills and spill_mask is not None:
            overlay = frame.copy()
            overlay[spill_mask > 0] = (255, 140, 0)  # Bright cyan/orange spill glow
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Draw Cradle Safe ROI Boundary
        if self.draw_cradle_roi:
            roi_pts = self.cradle_detector.get_roi_pixels(w, h)
            roi_color = (0, 255, 0) if overall == "SAFE" else (0, 0, 255)
            cv2.polylines(frame, [roi_pts], isClosed=True, color=roi_color, thickness=3)
            cv2.putText(frame, "SAFE CRADLE ZONE", (roi_pts[0][0] + 10, roi_pts[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # Draw Hazards (Spills & Dangerous Objects)
        for haz in hazards:
            hx1, hy1, hx2, hy2 = haz["bbox"]
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 165, 255), 2)
            cv2.putText(frame, f"HAZARD: {haz['label']}", (hx1, max(15, hy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Draw Children Bounding Boxes & Pose Skeleton
        for child in children:
            x1, y1, x2, y2 = child["bbox"]
            fall_res = child.get("fall_analysis", {})
            is_fall = fall_res.get("is_fall", False)

            # Box color: Red if fall, Orange if lying/warning, Green if safe standing
            box_color = (0, 0, 255) if is_fall else ((0, 165, 255) if fall_res.get("posture") == "Fallen / Lying" else (0, 255, 0))
            
            if self.draw_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                label = f"Child #{child['id']} | {fall_res.get('posture', 'Active')}"
                if is_fall:
                    label += " [FALL DETECTED!]"
                
                # Label backdrop box
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 9, y1), box_color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw Pose Keypoint Joints
            kpts = child.get("keypoints")
            if self.draw_skeletons and kpts is not None:
                for kpt in kpts:
                    kx, ky = int(kpt[0]), int(kpt[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(frame, (kx, ky), 4, (255, 255, 0), -1)

        # Draw Top Safety Banner
        banner_bg = (0, 180, 0) if overall == "SAFE" else ((0, 140, 255) if overall == "WARNING" else (0, 0, 220))
        cv2.rectangle(frame, (0, 0), (w, 40), banner_bg, -1)
        banner_text = f"STATUS: {overall} | Active Kids: {len(children)} | Falls: {self.stats['fall_events']} | Hazards: {len(hazards)}"
        cv2.putText(frame, banner_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
