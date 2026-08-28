import os
import cv2
import time
import shutil
import logging
import threading
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from vision.detector import ChildMonitoringEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChildMonitoringApp")

app = FastAPI(title="AI Child Monitoring & Safety System")

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("sample_data", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/sounds", exist_ok=True)

# Mount static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global Active Engine and Video Source Manager with Thread Safety Lock
class VideoStreamManager:
    def __init__(self):
        self.source_type = "file"  # "file", "webcam", "url"
        self.source_path = "sample_data/cradle_demo.mp4"
        self.cap = None
        self.engine = ChildMonitoringEngine(mode="all")
        self.is_running = True
        self.lock = threading.Lock()
        self._init_cap()

    def _init_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        logger.info(f"Initializing video capture: type={self.source_type}, path={self.source_path}")
        if self.source_type == "webcam":
            try:
                cam_idx = int(self.source_path) if str(self.source_path).isdigit() else 0
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            except Exception as e:
                logger.error(f"Failed to open webcam: {e}")
                self.cap = None
        else:
            self.cap = cv2.VideoCapture(self.source_path)

    def set_source(self, source_type: str, source_path: str):
        with self.lock:
            self.source_type = source_type
            self.source_path = source_path
            self._init_cap()
            # Reset counters when source changes
            self.engine.stats["fall_events"] = 0
            self.engine.stats["cradle_breaches"] = 0
            self.engine.stats["hazard_alerts"] = 0

    def get_frame(self):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self._init_cap()

            if self.cap is None or not self.cap.isOpened():
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "VIDEO FEED UNAVAILABLE", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(blank, f"Source: {self.source_type} ({self.source_path})", (120, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                time.sleep(0.1)
                return self.engine.process_frame(blank)

            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "END OF STREAM", (230, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    time.sleep(0.1)
                    return self.engine.process_frame(blank)

            return self.engine.process_frame(frame)

stream_manager = VideoStreamManager()

# --- Pydantic Data Models ---
class SourceModel(BaseModel):
    source_type: str  # "file", "webcam", "url"
    source_path: str  # filepath, index, or RTSP URL

class RoiModel(BaseModel):
    polygon: List[List[int]]

class OverlayToggleModel(BaseModel):
    draw_boxes: Optional[bool] = None
    draw_skeletons: Optional[bool] = None
    draw_cradle_roi: Optional[bool] = None
    draw_spills: Optional[bool] = None

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_mjpeg():
    while stream_manager.is_running:
        annotated_frame, _ = stream_manager.get_frame()
        # Encode BGR image to JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)  # approx 30 fps frame generation delay

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stats")
async def get_stats():
    return JSONResponse(stream_manager.engine.stats)

@app.post("/api/set_source")
async def set_source(data: SourceModel):
    stream_manager.set_source(data.source_type, data.source_path)
    return {"status": "success", "source_type": data.source_type, "source_path": data.source_path}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: MP4, AVI, MOV, MKV, WEBM")

    save_path = os.path.join("uploads", file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Immediately set as current stream source
    stream_manager.set_source("file", save_path)
    return {"status": "success", "filename": file.filename, "file_path": save_path}

@app.post("/api/set_mode")
async def set_mode(mode: str = Form(...)):
    if mode in ("cradle", "play_area", "all"):
        stream_manager.engine.mode = mode
        return {"status": "success", "mode": mode}
    raise HTTPException(status_code=400, detail="Invalid mode")

@app.post("/api/set_roi")
async def set_roi(data: RoiModel):
    stream_manager.engine.cradle_detector.set_custom_roi(data.polygon)
    return {"status": "success", "polygon": data.polygon}

@app.post("/api/toggle_overlays")
async def toggle_overlays(data: OverlayToggleModel):
    eng = stream_manager.engine
    if data.draw_boxes is not None:
        eng.draw_boxes = data.draw_boxes
    if data.draw_skeletons is not None:
        eng.draw_skeletons = data.draw_skeletons
    if data.draw_cradle_roi is not None:
        eng.draw_cradle_roi = data.draw_cradle_roi
    if data.draw_spills is not None:
        eng.draw_spills = data.draw_spills
    return {
        "status": "success",
        "draw_boxes": eng.draw_boxes,
        "draw_skeletons": eng.draw_skeletons,
        "draw_cradle_roi": eng.draw_cradle_roi,
        "draw_spills": eng.draw_spills
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
