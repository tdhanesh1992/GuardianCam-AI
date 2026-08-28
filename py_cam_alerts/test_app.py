import cv2
import numpy as np
import time
from vision.detector import ChildMonitoringEngine
from vision.fall_detector import FallDetector
from vision.cradle_detector import CradleDetector
from vision.hazard_detector import HazardDetector

def test_vision_engine():
    print("Testing Vision Engine components...")
    engine = ChildMonitoringEngine(mode="all")

    # Read frame from generated sample video
    cap = cv2.VideoCapture("sample_data/fall_demo.mp4")
    assert cap.isOpened(), "Failed to open sample_data/fall_demo.mp4"

    ret, frame = cap.read()
    assert ret and frame is not None, "Failed to read frame"

    # Process frame through vision engine
    annotated, analysis = engine.process_frame(frame)
    assert annotated is not None, "Annotated frame is None"
    assert "stats" in analysis, "Analysis missing stats key"
    assert "children" in analysis, "Analysis missing children key"

    print("Vision Engine processing passed successfully!")
    print("Stats:", analysis['stats'])
    cap.release()

if __name__ == "__main__":
    test_vision_engine()
