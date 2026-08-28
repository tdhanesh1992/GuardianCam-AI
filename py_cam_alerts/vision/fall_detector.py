import time
import math
import numpy as np

class FallDetector:
    """
    Fall Detection Engine analyzing posture keypoints, aspect ratios, angles,
    and velocity history to identify child falls in real-time.
    """
    def __init__(self, history_len=15):
        self.history_len = history_len
        # Stores track_id -> list of (timestamp, centroid_y, head_y, is_lying)
        self.track_history = {}
        # Stores active fall alert states to avoid rapid flickering
        self.alert_states = {}

    def analyze_pose(self, person_id, bbox, keypoints=None):
        """
        Analyze bounding box and keypoints for a person.
        bbox: [x1, y1, x2, y2]
        keypoints: ndarray of shape (17, 2 or 3) [x, y, conf]
        returns dict with:
          is_fall (bool), confidence (float), posture (str), details (dict)
        """
        now = time.time()
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = width / float(height)
        centroid_y = (y1 + y2) / 2.0
        centroid_x = (x1 + x2) / 2.0

        is_lying = False
        torso_angle = 0.0
        head_hip_rel = 0.0
        pose_confidence = 0.5

        if keypoints is not None and len(keypoints) >= 17:
            # COCO Keypoint mapping:
            # 0: nose, 1: L eye, 2: R eye, 3: L ear, 4: R ear
            # 5: L shoulder, 6: R shoulder, 7: L elbow, 8: R elbow
            # 9: L wrist, 10: R wrist, 11: L hip, 12: R hip
            # 13: L knee, 14: R knee, 15: L ankle, 16: R ankle
            
            # Extract key joints
            nose = keypoints[0][:2]
            l_sh = keypoints[5][:2]
            r_sh = keypoints[6][:2]
            l_hip = keypoints[11][:2]
            r_hip = keypoints[12][:2]

            sh_center = (l_sh + r_sh) / 2.0 if (np.any(l_sh) and np.any(r_sh)) else l_sh
            hip_center = (l_hip + r_hip) / 2.0 if (np.any(l_hip) and np.any(r_hip)) else l_hip

            # Check if torso vector exists
            if np.any(sh_center) and np.any(hip_center):
                dx = sh_center[0] - hip_center[0]
                dy = sh_center[1] - hip_center[1]  # positive Y is downwards in images
                
                # Torso angle relative to vertical axis (0 deg is upright, 90 deg is horizontal)
                torso_angle = abs(math.degrees(math.atan2(dx, -dy)))
                if torso_angle > 90:
                    torso_angle = 180 - torso_angle

            # Check head vs hip Y level
            head_y = nose[1] if np.any(nose) else y1
            head_hip_rel = hip_center[1] - head_y  # positive if head is above hips

            # Pose horizontal condition
            if torso_angle > 55.0 or aspect_ratio > 1.1 or head_hip_rel < height * 0.15:
                is_lying = True
        else:
            # Fallback to Bounding Box ratio analysis
            if aspect_ratio > 1.15:
                is_lying = True
                torso_angle = 70.0

        # Maintain track history for velocity calculation
        if person_id not in self.track_history:
            self.track_history[person_id] = []

        history = self.track_history[person_id]
        history.append((now, centroid_y, head_y if keypoints is not None else y1, is_lying))
        if len(history) > self.history_len:
            history.pop(0)

        # Calculate vertical velocity over recent frames
        velocity = 0.0
        sudden_drop = False
        if len(history) >= 3:
            dt = history[-1][0] - history[0][0]
            dy = history[-1][1] - history[0][1]
            if dt > 0.01:
                velocity = dy / dt  # pixels per second drop

            # Check for sudden drop in past 0.5s
            recent_dy = history[-1][1] - history[-3][1]
            if recent_dy > (height * 0.35):
                sudden_drop = True

        # Fall Scoring heuristic
        fall_score = 0.0
        if is_lying:
            fall_score += 0.5
        if torso_angle > 60:
            fall_score += 0.25
        if sudden_drop or velocity > (height * 1.5):
            fall_score += 0.35
        if aspect_ratio > 1.25:
            fall_score += 0.15

        is_fall = fall_score >= 0.60
        
        # Smooth alert output (hysteresis)
        if person_id not in self.alert_states:
            self.alert_states[person_id] = {"active": False, "since": 0}

        st = self.alert_states[person_id]
        if is_fall:
            if not st["active"]:
                st["active"] = True
                st["since"] = now
        else:
            # Clear fall state if upright for at least 1.5 seconds
            if st["active"] and (now - st["since"] > 1.5):
                st["active"] = False

        posture_str = "Fallen / Lying" if is_lying else ("Sitting/Squatting" if aspect_ratio > 0.8 else "Standing/Walking")

        return {
            "is_fall": st["active"],
            "fall_score": min(1.0, round(fall_score, 2)),
            "posture": posture_str,
            "aspect_ratio": round(aspect_ratio, 2),
            "torso_angle_deg": round(torso_angle, 1),
            "sudden_drop": sudden_drop,
            "velocity": round(velocity, 1)
        }
