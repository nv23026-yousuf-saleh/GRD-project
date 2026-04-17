"""
Plithos — FastAPI Backend  (accuracy-fixed)
--------------------------------------------
Key fixes vs original:
  1. Thresholds match app.py exactly (no more lower-threshold false positives)
  2. Per-connection InferenceSession: each WebSocket client gets its own
     rolling-mean buffer so clients don't pollute each other's state
  3. Scene validation applied before inference — dark/blank frames return
     {skipped: true} instead of spurious violence scores
  4. MJPEG /stream no longer calls _run_inference() on every encode cycle;
     it uses a shared annotated-frame buffer updated by one background thread
  5. /detect/frame and /detect/image share the same stateful session object
     so the rolling mean accumulates properly across HTTP calls

Changes:
  - Violence detection only triggers when >= 2 persons are in frame
  - Smoking detection uses a custom primary model with an optional verifier

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import cv2
import numpy as np
import base64
import time
import threading
import queue
import json
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── Import detection logic ────────────────────────────────────────────────────
import importlib.util, sys, os

_spec = importlib.util.spec_from_file_location(
    "detection", os.path.join(os.path.dirname(__file__), "app.py"))
_det = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_det)

preprocess_frame_violence = _det.preprocess_frame_violence
predict_violence          = _det.predict_violence
predict_litter            = _det.predict_litter
predict_smoking           = _det.predict_smoking
predict_fire              = _det.predict_fire
predict_persons           = _det.predict_persons
draw_overlay              = _det.draw_overlay
load_litter_model         = _det.load_litter_model
load_smoking_model        = _det.load_smoking_model
load_violence_model       = _det.load_violence_model
load_fire_model           = _det.load_fire_model
load_person_model         = _det.load_person_model
is_valid_frame            = _det.is_valid_frame

# ── Config: mirror app.py exactly so thresholds are never out of sync ────────
VIOLENCE_H5_PATH     = _det.VIOLENCE_H5_PATH
VIOLENCE_ONNX_PATH   = _det.VIOLENCE_ONNX_PATH
SMOKING_MODEL_PATH   = _det.SMOKING_MODEL_PATH
SMOKING_VERIFY_MODEL_PATH = _det.SMOKING_VERIFY_MODEL_PATH
FIRE_MODEL_PATH      = _det.FIRE_MODEL_PATH
VIOLENCE_THRESHOLD   = _det.VIOLENCE_THRESHOLD    # 0.90
VIOLENCE_SMOOTH      = _det.VIOLENCE_SMOOTH        # 5
VIOLENCE_MIN_PERSONS = _det.VIOLENCE_MIN_PERSONS   # 2
VIOLENCE_PERSON_GRACE_SEC = _det.VIOLENCE_PERSON_GRACE_SEC
SMOKING_THRESHOLD    = _det.SMOKING_THRESHOLD      # 0.55
FIRE_THRESHOLD       = _det.FIRE_THRESHOLD         # 0.55
YOLO_IMGSZ           = _det.YOLO_IMGSZ
MAX_ALERT_HISTORY    = 100

# Propagate to module so all predict_* functions use the same values
_det.VIOLENCE_THRESHOLD   = VIOLENCE_THRESHOLD
_det.SMOKING_THRESHOLD    = SMOKING_THRESHOLD
_det.LITTER_THRESHOLD     = SMOKING_THRESHOLD
_det.FIRE_THRESHOLD       = FIRE_THRESHOLD
_det.YOLO_IMGSZ           = YOLO_IMGSZ
_det.MODE                 = "both"

# Legacy aliases kept for older callers.
LITTER_MODEL_PATH = SMOKING_MODEL_PATH
LITTER_VERIFY_MODEL_PATH = SMOKING_VERIFY_MODEL_PATH
LITTER_THRESHOLD = SMOKING_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Plithos Detection API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# STATEFUL INFERENCE SESSION
# One per logical "camera stream".  Holds the rolling-mean buffer so
# predictions smooth correctly across sequential HTTP calls.
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSession:
    """
    Thread-safe inference session with rolling-mean violence smoothing.

    Violence person-gate: violence is suppressed unless >= VIOLENCE_MIN_PERSONS
    people are detected in the frame, preventing false alerts on lone individuals
    or empty scenes.

    Smoking detection uses the shared custom-model pipeline from app.py.
    """

    def __init__(self, v_model, l_model, f_model, p_model):
        self.v_model   = v_model
        self.l_model   = l_model
        self.f_model   = f_model
        self.p_model   = p_model
        self.lock      = threading.Lock()
        self.frame_buf = deque(maxlen=max(_det.SEQUENCE_LEN, 1))
        self.conf_hist = deque(maxlen=VIOLENCE_SMOOTH)
        self.violence_person_gate = _det.ViolencePersonGate(
            VIOLENCE_MIN_PERSONS, VIOLENCE_PERSON_GRACE_SEC)

    def infer(self, frame: np.ndarray) -> Dict[str, Any]:
        valid, reason = is_valid_frame(frame)
        if not valid:
            return {
                "skipped": True,
                "skip_reason": reason,
                "is_violent": False,
                "violence_confidence": 0.0,
                "is_smoking": False,
                "smoking_detections": [],
                "is_littering": False,
                "litter_detections": [],
                "is_fire": False,
                "is_smoke": False,
                "fire_detections": [],
                "person_count": 0,
                "timestamp": datetime.now().isoformat(),
            }

        with self.lock:
            # Persons — run first so person-gate is available for violence check
            if self.p_model:
                person_count, person_detections = predict_persons(self.p_model, frame)
            else:
                person_count = 0
                person_detections = []

            gate_ready, gate_remaining = self.violence_person_gate.update(person_count)

            # Violence — only evaluate after the crowd gate and grace period pass
            if self.v_model:
                self.frame_buf.append(preprocess_frame_violence(frame))
                raw_is_violent, raw_v_conf = predict_violence(
                    self.v_model, self.frame_buf, self.conf_hist)
                if gate_ready:
                    is_violent, v_conf = raw_is_violent, raw_v_conf
                else:
                    is_violent, v_conf = False, 0.0
            else:
                is_violent, v_conf = False, 0.0
                gate_remaining = VIOLENCE_PERSON_GRACE_SEC

            # Smoking detection
            if self.l_model:
                is_smoking, smoking_dets = predict_smoking(self.l_model, frame, person_detections)
            else:
                is_smoking, smoking_dets = False, []

            # Fire
            if self.f_model:
                is_fire, is_smoke, fire_dets = predict_fire(self.f_model, frame)
            else:
                is_fire, is_smoke, fire_dets = False, False, []

        ts = datetime.now().isoformat()
        return {
            "skipped": False,
            "is_violent":           bool(is_violent),
            "violence_confidence":  round(float(v_conf), 4),
            "is_smoking":           bool(is_smoking),
            "smoking_detections":   [
                {"label": d["label"],
                 "confidence": round(float(d["conf"]), 4),
                 "bbox": list(d["bbox"])}
                for d in smoking_dets
            ],
            "is_littering":         bool(is_smoking),
            "litter_detections":    [
                {"label": d["label"],
                 "confidence": round(float(d["conf"]), 4),
                 "bbox": list(d["bbox"])}
                for d in smoking_dets
            ],
            "is_fire":   bool(is_fire),
            "is_smoke":  bool(is_smoke),
            "fire_detections": [
                {"label": d["label"],
                 "confidence": round(float(d["conf"]), 4),
                 "bbox": list(d["bbox"])}
                for d in fire_dets
            ],
            "person_count": person_count,
            "violence_gate_ready": gate_ready,
            "violence_gate_remaining": round(float(gate_remaining), 3),
            "timestamp": ts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL APP STATE
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    v_model      = None
    l_model      = None
    f_model      = None
    p_model      = None
    models_ready = False
    http_session: Optional[InferenceSession] = None
    alert_history: List[Dict[str, Any]] = []
    last_result: Dict[str, Any] = {
        "is_violent": False,
        "violence_confidence": 0.0,
        "is_smoking": False,
        "smoking_detections": [],
        "is_littering": False,
        "litter_detections": [],
        "is_fire": False,
        "is_smoke": False,
        "timestamp": None,
    }

state = AppState()


@app.on_event("startup")
async def startup_event():
    def _load():
        try:
            print("[API] Loading models ...")
            state.v_model = load_violence_model(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH)
            state.l_model = load_smoking_model(SMOKING_MODEL_PATH, SMOKING_VERIFY_MODEL_PATH)
            try:
                state.f_model = load_fire_model(FIRE_MODEL_PATH)
            except FileNotFoundError:
                print("[API] Fire model not found — fire detection disabled")
                state.f_model = None
            state.p_model = load_person_model()
            state.http_session = InferenceSession(
                state.v_model, state.l_model, state.f_model, state.p_model)
            state.models_ready = True
            print("[API] Models loaded — ready ✓")
            print(f"[API] Thresholds: violence={VIOLENCE_THRESHOLD}  "
                  f"smoking={SMOKING_THRESHOLD}  fire={FIRE_THRESHOLD}")
            print(f"[API] Violence min persons: {VIOLENCE_MIN_PERSONS}")
            print(f"[API] Violence grace period: {VIOLENCE_PERSON_GRACE_SEC:.1f}s")
            print(f"[API] Smoking primary model: {SMOKING_MODEL_PATH}")
            print(f"[API] Smoking verifier model: {SMOKING_VERIFY_MODEL_PATH}")
        except Exception as e:
            print(f"[API] WARNING: Could not load models: {e}")

    threading.Thread(target=_load, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _require_models():
    if not state.models_ready:
        raise HTTPException(status_code=503, detail="Models not ready. Try again shortly.")


def _decode_image(data: bytes) -> np.ndarray:
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def _decode_base64_frame(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    return _decode_image(base64.b64decode(b64))


def _record_alert(result: Dict[str, Any]):
    state.last_result = result
    if result.get("is_violent") or result.get("is_smoking") or result.get("is_littering") or \
       result.get("is_fire") or result.get("is_smoke"):
        alert = {
            "type":      ("violence" if result["is_violent"]
                          else "fire" if result.get("is_fire")
                          else "smoke" if result.get("is_smoke")
                          else "smoking"),
            "v_conf":    result.get("violence_confidence", 0),
            "objects":   len(result.get("smoking_detections", result.get("litter_detections", []))),
            "timestamp": result["timestamp"],
        }
        state.alert_history.insert(0, alert)
        if len(state.alert_history) > MAX_ALERT_HISTORY:
            state.alert_history.pop()


def _annotate_frame(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    smoking_dets = [
        {"bbox": tuple(d["bbox"]), "conf": d["confidence"], "label": d["label"]}
        for d in result.get("smoking_detections", result.get("litter_detections", []))
    ]
    fire_dets = [
        {"bbox": tuple(d["bbox"]), "conf": d["confidence"], "label": d["label"]}
        for d in result.get("fire_detections", [])
    ]
    return draw_overlay(
        frame,
        result.get("is_violent", False),
        result.get("violence_confidence", 0.0),
        result.get("is_smoking", result.get("is_littering", False)),
        smoking_dets,
        result.get("is_fire", False),
        result.get("is_smoke", False),
        fire_dets,
        result.get("person_count", 0),
        [],
        fps_display=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    return {
        "status":       "ok",
        "models_ready": state.models_ready,
        "last_result":  state.last_result,
        "alert_count":  len(state.alert_history),
        "thresholds": {
            "violence":            VIOLENCE_THRESHOLD,
            "violence_min_persons": VIOLENCE_MIN_PERSONS,
            "violence_person_grace_sec": VIOLENCE_PERSON_GRACE_SEC,
            "smoking":             SMOKING_THRESHOLD,
            "fire":                FIRE_THRESHOLD,
        },
    }


@app.get("/alerts")
def get_alerts(limit: int = 50):
    return {"alerts": state.alert_history[:limit]}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Upload JPEG/PNG → returns detection JSON."""
    _require_models()
    raw    = await file.read()
    frame  = _decode_image(raw)
    result = state.http_session.infer(frame)
    _record_alert(result)
    return result


class FrameRequest(BaseModel):
    frame:    str
    annotate: bool = False


@app.post("/detect/frame")
async def detect_frame(req: FrameRequest):
    """
    Send a base64 webcam frame → returns detection JSON.
    The stateful http_session accumulates rolling-mean history across calls,
    matching the behaviour of the standalone app.
    """
    _require_models()
    frame  = _decode_base64_frame(req.frame)
    result = state.http_session.infer(frame)
    _record_alert(result)

    if req.annotate and not result.get("skipped"):
        annotated = _annotate_frame(frame.copy(), result)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result["annotated_frame"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — per-connection session (no shared state between clients)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Each WebSocket connection gets its own InferenceSession so rolling-mean
    buffers don't mix between multiple browser tabs / clients.

    Protocol:
      Client → Server:  JSON { "frame": "<base64>", "annotate": bool }
      Server → Client:  JSON { ...detection result... }
    """
    await ws.accept()

    if not state.models_ready:
        await ws.send_text(json.dumps({"error": "Models not ready yet."}))
        await ws.close()
        return

    # Per-connection stateful session
    session = InferenceSession(
        state.v_model, state.l_model, state.f_model, state.p_model)

    try:
        while True:
            text    = await ws.receive_text()
            payload = json.loads(text)
            b64     = payload.get("frame", "")
            if not b64:
                await ws.send_text(json.dumps({"error": "no frame provided"}))
                continue

            try:
                frame  = _decode_base64_frame(b64)
                result = session.infer(frame)
                _record_alert(result)

                if payload.get("annotate") and not result.get("skipped"):
                    annotated = _annotate_frame(frame.copy(), result)
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    result["annotated_frame"] = (
                        "data:image/jpeg;base64," + base64.b64encode(buf).decode())

                await ws.send_text(json.dumps(result))
            except Exception as e:
                await ws.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG STREAM — single background inference thread, not per-frame re-inference
# ─────────────────────────────────────────────────────────────────────────────

_stream_state = {
    "cap":       None,
    "thread":    None,
    "frame":     None,
    "lock":      threading.Lock(),
    "running":   False,
}


def _stream_worker():
    """Background thread: capture → infer → annotate → store bytes."""
    session = InferenceSession(
        state.v_model, state.l_model, state.f_model, state.p_model)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _stream_state["running"] = True

    while _stream_state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        result    = session.infer(frame)
        annotated = _annotate_frame(frame, result) if not result.get("skipped") else frame
        _, buf    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])

        with _stream_state["lock"]:
            _stream_state["frame"] = buf.tobytes()

        time.sleep(0.033)

    cap.release()


def _ensure_stream_worker():
    if _stream_state["thread"] is None or not _stream_state["thread"].is_alive():
        t = threading.Thread(target=_stream_worker, daemon=True)
        t.start()
        _stream_state["thread"] = t


def _gen_mjpeg():
    _ensure_stream_worker()
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initialising...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    _, pbuf = cv2.imencode(".jpg", placeholder)
    ph_bytes = pbuf.tobytes()

    while True:
        with _stream_state["lock"]:
            frame_bytes = _stream_state["frame"]
        payload = frame_bytes if frame_bytes else ph_bytes
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
        time.sleep(0.033)


@app.get("/stream")
def mjpeg_stream():
    """MJPEG annotated webcam stream.  Use: <img src='http://localhost:8000/stream'>"""
    if not state.models_ready:
        raise HTTPException(status_code=503, detail="Models not ready.")
    return StreamingResponse(_gen_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
