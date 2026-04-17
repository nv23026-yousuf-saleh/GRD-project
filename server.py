from __future__ import annotations

import argparse
import csv
import importlib
import io
import os
import smtplib
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import wraps
from typing import Any

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(__file__))
app_module = importlib.import_module("app")

try:
    api_module = importlib.import_module("api")
    InferenceSession = api_module.InferenceSession
    HAS_API = True
except Exception as exc:  # pragma: no cover - import depends on optional local env
    InferenceSession = None
    HAS_API = False
    API_IMPORT_ERROR = str(exc)

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("PLITHOS_DB_PATH", os.path.join(HERE, "plithos.db"))
JPEG_QUALITY = 64
ACTIVE_SLEEP = 0.01
IDLE_SLEEP = 0.12
EMAIL_RATE_LIMIT_SECONDS = 60
CAMERA_SCAN_MAX = int(os.environ.get("PLITHOS_CAMERA_SCAN_MAX", "6") or 6)
CAMERA_SCAN_CACHE_SECONDS = 8.0
PREVIEW_CACHE_SECONDS = 1.0
MAX_INFERENCE_FPS = 20.0
MAX_RECENT_ALERTS = 50
MAX_LOG_ROWS = 250
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 360
TARGET_CAMERA_FPS = 30

flask_app = Flask(__name__)
flask_app.secret_key = os.environ.get("PLITHOS_SECRET_KEY", "plithos-dev-secret")
flask_app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

registry_lock = threading.Lock()
camera_runtimes: dict[int, "CameraRuntime"] = {}
camera_scan_lock = threading.Lock()
camera_scan_cache: dict[str, Any] = {"checked_at": 0.0, "items": []}
preview_cache_lock = threading.Lock()
preview_frame_cache: dict[str, dict[str, Any]] = {}


def now_local() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def human_time(value: str | None) -> str:
    parsed = parse_iso(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else "--"


def db_connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                theme TEXT NOT NULL DEFAULT 'dark',
                sound_enabled INTEGER NOT NULL DEFAULT 1,
                auto_switch_alerts INTEGER NOT NULL DEFAULT 1,
                notifications_enabled INTEGER NOT NULL DEFAULT 1,
                smtp_host TEXT DEFAULT '',
                smtp_port INTEGER NOT NULL DEFAULT 587,
                smtp_username TEXT DEFAULT '',
                smtp_password TEXT DEFAULT '',
                smtp_sender TEXT DEFAULT '',
                smtp_use_tls INTEGER NOT NULL DEFAULT 1,
                report_email TEXT DEFAULT '',
                alert_subject TEXT DEFAULT '',
                alert_message TEXT DEFAULT '',
                last_email_sent_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                camera_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                detail TEXT NOT NULL,
                confidence REAL,
                persons INTEGER,
                created_at TEXT NOT NULL
            );
            """
        )
        existing_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        required_columns = {
            "report_email": "TEXT DEFAULT ''",
            "alert_subject": "TEXT DEFAULT ''",
            "alert_message": "TEXT DEFAULT ''",
        }
        for column, ddl in required_columns.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} {ddl}")
        conn.execute(
            """
            UPDATE users
            SET report_email = COALESCE(NULLIF(report_email, ''), email),
                alert_subject = COALESCE(alert_subject, ''),
                alert_message = COALESCE(alert_message, '')
            """
        )
        conn.execute("UPDATE alerts SET event_type = 'smoking' WHERE event_type = 'littering'")
        conn.commit()


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with db_connect() as conn:
        return conn.execute(query, params).fetchone()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with db_connect() as conn:
        return conn.execute(query, params).fetchall()


def execute_db(query: str, params: tuple[Any, ...] = ()) -> int:
    with db_connect() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid


def count_users() -> int:
    row = fetch_one("SELECT COUNT(*) AS count FROM users")
    return int(row["count"]) if row else 0


def get_user_by_id(user_id: int | None) -> sqlite3.Row | None:
    if not user_id:
        return None
    return fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))


def get_user_by_email(email: str) -> sqlite3.Row | None:
    return fetch_one("SELECT * FROM users WHERE lower(email) = lower(?)", (email.strip(),))


def cameras_for_user(user_id: int) -> list[sqlite3.Row]:
    return fetch_all(
        "SELECT * FROM cameras WHERE user_id = ? AND enabled = 1 ORDER BY sort_order, id",
        (user_id,),
    )


def user_has_cameras(user_id: int) -> bool:
    row = fetch_one("SELECT COUNT(*) AS count FROM cameras WHERE user_id = ? AND enabled = 1", (user_id,))
    return bool(row and row["count"])


def alerts_for_user(
    user_id: int,
    *,
    limit: int = MAX_RECENT_ALERTS,
    camera_id: int | None = None,
    event_type: str | None = None,
    search: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[sqlite3.Row]:
    clauses = ["user_id = ?"]
    params: list[Any] = [user_id]
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if event_type and event_type != "all":
        clauses.append("event_type = ?")
        params.append(event_type)
    if search:
        clauses.append("(camera_name LIKE ? OR detail LIKE ?)")
        token = f"%{search.strip()}%"
        params.extend([token, token])
    if date_from:
        clauses.append("created_at >= ?")
        params.append(f"{date_from}T00:00:00")
    if date_to:
        clauses.append("created_at <= ?")
        params.append(f"{date_to}T23:59:59")
    query = f"""
        SELECT * FROM alerts
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC
        LIMIT ?
    """
    params.append(limit)
    return fetch_all(query, tuple(params))


def total_alerts_for_user(user_id: int) -> int:
    row = fetch_one("SELECT COUNT(*) AS count FROM alerts WHERE user_id = ?", (user_id,))
    return int(row["count"]) if row else 0


def validate_email(value: str) -> bool:
    return bool(value and "@" in value and "." in value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def default_alert_subject() -> str:
    return "Plithos alert"


def default_alert_message() -> str:
    return "Plithos detected an incident. A report is attached."


def report_recipient(user: sqlite3.Row) -> str:
    return str(user["report_email"] or user["email"] or "").strip().lower()


def mail_delivery_config(user: sqlite3.Row | None = None) -> dict[str, Any]:
    host = os.environ.get("PLITHOS_EMAIL_HOST", "").strip()
    port = os.environ.get("PLITHOS_EMAIL_PORT", "").strip() or "587"
    username = os.environ.get("PLITHOS_EMAIL_USERNAME", "").strip()
    password = os.environ.get("PLITHOS_EMAIL_PASSWORD", "")
    sender = os.environ.get("PLITHOS_EMAIL_SENDER", "").strip()
    if host and sender:
        return {
            "host": host,
            "port": int(port or 587),
            "username": username,
            "password": password,
            "sender": sender,
            "use_tls": env_flag("PLITHOS_EMAIL_USE_TLS", True),
        }
    if user and user["smtp_host"] and user["smtp_sender"]:
        return {
            "host": str(user["smtp_host"]).strip(),
            "port": int(user["smtp_port"] or 587),
            "username": str(user["smtp_username"] or "").strip(),
            "password": str(user["smtp_password"] or ""),
            "sender": str(user["smtp_sender"]).strip(),
            "use_tls": bool(user["smtp_use_tls"]),
        }
    return {"host": "", "port": 587, "username": "", "password": "", "sender": "", "use_tls": True}


def email_sender_ready(user: sqlite3.Row | None = None) -> bool:
    config = mail_delivery_config(user)
    return bool(config["host"] and config["sender"])


def send_email_message(
    *,
    to_address: str,
    subject: str,
    body: str,
    user: sqlite3.Row | None = None,
    attachment_name: str | None = None,
    attachment_bytes: bytes | None = None,
) -> tuple[bool, str]:
    config = mail_delivery_config(user)
    if not config["host"] or not config["sender"] or not to_address:
        return False, "Email delivery is not configured."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["sender"]
    message["To"] = to_address
    message.set_content(body)
    if attachment_name and attachment_bytes is not None:
        message.add_attachment(
            attachment_bytes,
            maintype="text",
            subtype="csv",
            filename=attachment_name,
        )

    try:
        with smtplib.SMTP(config["host"], int(config["port"] or 587), timeout=20) as smtp:
            if config["use_tls"]:
                smtp.starttls()
            if config["username"]:
                smtp.login(config["username"], config["password"] or "")
            smtp.send_message(message)
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on local mail configuration
        return False, str(exc)


def create_user(full_name: str, email: str, password: str) -> int:
    return execute_db(
        """
        INSERT INTO users (full_name, email, password_hash, report_email, alert_subject, alert_message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            full_name.strip(),
            email.strip().lower(),
            generate_password_hash(password),
            email.strip().lower(),
            default_alert_subject(),
            default_alert_message(),
            now_iso(),
        ),
    )


def update_user_settings(user_id: int, payload: dict[str, Any]) -> None:
    fields = ", ".join(f"{key} = ?" for key in payload.keys())
    execute_params = tuple(payload.values()) + (user_id,)
    with db_connect() as conn:
        conn.execute(f"UPDATE users SET {fields} WHERE id = ?", execute_params)
        conn.commit()


def replace_cameras(user_id: int, cameras: list[dict[str, str]]) -> None:
    with db_connect() as conn:
        existing = conn.execute(
            "SELECT id FROM cameras WHERE user_id = ? AND enabled = 1 ORDER BY sort_order, id",
            (user_id,),
        ).fetchall()
        created_at = now_iso()
        for index, camera in enumerate(cameras):
            if index < len(existing):
                conn.execute(
                    """
                    UPDATE cameras
                    SET name = ?, source = ?, enabled = 1, sort_order = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        camera["name"].strip(),
                        camera["source"].strip(),
                        index,
                        created_at,
                        existing[index]["id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO cameras (user_id, name, source, enabled, sort_order, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        user_id,
                        camera["name"].strip(),
                        camera["source"].strip(),
                        index,
                        created_at,
                        created_at,
                    ),
                )
        for offset, row in enumerate(existing[len(cameras):], start=len(cameras)):
            conn.execute(
                "UPDATE cameras SET enabled = 0, sort_order = ?, updated_at = ? WHERE id = ?",
                (offset, created_at, row["id"]),
            )
        conn.commit()


def alert_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "camera_name": row["camera_name"],
        "type": row["event_type"],
        "severity": row["severity"],
        "detail": row["detail"],
        "confidence": row["confidence"],
        "persons": row["persons"],
        "time": row["created_at"],
    }


def record_alert(
    *,
    user_id: int,
    camera_id: int,
    camera_name: str,
    event_type: str,
    severity: str,
    detail: str,
    confidence: float | None,
    persons: int | None,
) -> dict[str, Any]:
    created_at = now_iso()
    alert_id = execute_db(
        """
        INSERT INTO alerts (user_id, camera_id, camera_name, event_type, severity, detail, confidence, persons, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, camera_id, camera_name, event_type, severity, detail, confidence, persons, created_at),
    )
    alert = {
        "id": alert_id,
        "user_id": user_id,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "type": event_type,
        "severity": severity,
        "detail": detail,
        "confidence": confidence,
        "persons": persons,
        "time": created_at,
    }
    threading.Thread(target=send_alert_email_if_due, args=(alert,), daemon=True).start()
    return alert


def build_alert_csv(alert: dict[str, Any]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Time", "Camera", "Type", "Severity", "Detail", "Confidence", "Persons"])
    confidence = ""
    if alert.get("confidence") is not None:
        confidence = f"{float(alert['confidence']) * 100:.1f}%" if alert["type"] == "violence" else f"{float(alert['confidence']):.2f}"
    writer.writerow(
        [
            human_time(alert["time"]),
            alert["camera_name"],
            alert["type"],
            alert["severity"],
            alert["detail"],
            confidence,
            alert.get("persons") or "",
        ]
    )
    return stream.getvalue().encode("utf-8")


def send_alert_email_if_due(alert: dict[str, Any]) -> None:
    user = get_user_by_id(alert["user_id"])
    if not user or not user["notifications_enabled"]:
        return

    last_sent = parse_iso(user["last_email_sent_at"])
    if last_sent and (now_local() - last_sent) < timedelta(seconds=EMAIL_RATE_LIMIT_SECONDS):
        return

    recipient = report_recipient(user)
    if not recipient or not validate_email(recipient):
        return

    subject_base = str(user["alert_subject"] or "").strip() or default_alert_subject()
    subject = f"{subject_base} | {alert['camera_name']} | {alert['type'].title()}"
    confidence = ""
    if alert.get("confidence") is not None:
        confidence = f"{float(alert['confidence']) * 100:.1f}%" if alert["type"] == "violence" else f"{float(alert['confidence']):.2f}"
    intro = str(user["alert_message"] or "").strip() or default_alert_message()
    body = "\n".join(
        [
            intro,
            "",
            f"Time: {human_time(alert['time'])}",
            f"Camera: {alert['camera_name']}",
            f"Type: {alert['type'].title()}",
            f"Severity: {alert['severity']}",
            f"Detail: {alert['detail']}",
            f"Confidence: {confidence or '--'}",
            f"Persons: {alert.get('persons') or '--'}",
            "",
            "A CSV report is attached to this email.",
        ]
    )
    ok, error = send_email_message(
        to_address=recipient,
        subject=subject,
        body=body,
        user=user,
        attachment_name=f"plithos-alert-{alert['camera_id']}-{alert['id']}.csv",
        attachment_bytes=build_alert_csv(alert),
    )
    if ok:
        update_user_settings(alert["user_id"], {"last_email_sent_at": now_iso()})
    elif error:
        print(f"[WARN] Email send failed: {error}")


class SharedModels:
    def __init__(self) -> None:
        self.ready = False
        self.loading = False
        self.error = ""
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.v_model = None
        self.l_model = None
        self.f_model = None
        self.p_model = None

    def ensure_loaded(self) -> None:
        with self.lock:
            if self.ready or self.loading:
                return
            self.loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            run_v = app_module.MODE in ("both", "violence")
            run_l = app_module.MODE in ("both", "smoking", "litter")
            self.v_model = app_module.load_violence_model(app_module.VIOLENCE_H5_PATH, app_module.VIOLENCE_ONNX_PATH) if run_v else None
            self.l_model = app_module.load_smoking_model(app_module.SMOKING_MODEL_PATH, app_module.SMOKING_VERIFY_MODEL_PATH) if run_l else None
            try:
                self.f_model = app_module.load_fire_model(app_module.FIRE_MODEL_PATH) if app_module.ENABLE_FIRE else None
            except FileNotFoundError:
                print("[WARN] Fire model not found - fire detection disabled")
                self.f_model = None
            self.p_model = app_module.load_person_model()
            app_module.warmup(self.v_model, self.l_model, self.f_model, self.p_model)
            self.ready = True
            self.error = ""
        except Exception as exc:  # pragma: no cover - model env specific
            self.error = str(exc)
            print(f"[ERROR] Model loading failed: {exc}")
        finally:
            self.loading = False


shared_models = SharedModels()


def base_camera_state(camera_id: int, name: str, source: str, user_id: int) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "user_id": user_id,
        "name": name,
        "source": source,
        "camera_online": False,
        "models_ready": shared_models.ready,
        "frame_skipped": False,
        "is_violent": False,
        "v_conf": 0.0,
        "is_smoking": False,
        "smoking_dets": [],
        "is_fire": False,
        "is_smoke": False,
        "fire_dets": [],
        "person_count": 0,
        "fps": 0.0,
        "frame": 0,
        "video_timestamp": "",
        "updated_at": "",
        "uptime_sec": 0,
        "active_alert": False,
        "alert_type": "",
        "last_alert": None,
        "last_alert_id": None,
        "last_detail": "",
    }


def open_capture(source: str):
    src: Any = int(source) if str(source).isdigit() else source
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    return cap


def configure_capture(cap) -> None:
    if cap is None:
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass


def placeholder_frame(name: str, message: str) -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (26, 26), (614, 454), (44, 78, 112), 2)
    cv2.putText(img, name[:28], (52, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (236, 242, 247), 2, cv2.LINE_AA)
    cv2.putText(img, message, (52, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (162, 182, 201), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def discover_local_camera_sources(force: bool = False) -> list[dict[str, str]]:
    now_tick = time.monotonic()
    with camera_scan_lock:
        cached_items = camera_scan_cache.get("items", [])
        checked_at = float(camera_scan_cache.get("checked_at", 0.0))
        if not force and cached_items and (now_tick - checked_at) < CAMERA_SCAN_CACHE_SECONDS:
            return [dict(item) for item in cached_items]

    discovered: list[dict[str, str]] = []
    for index in range(max(CAMERA_SCAN_MAX, 1)):
        cap = open_capture(str(index))
        try:
            if not cap.isOpened():
                continue
            configure_capture(cap)
            ok, _ = cap.read()
            if ok:
                discovered.append({"source": str(index), "label": f"Camera {len(discovered) + 1}"})
        finally:
            cap.release()

    with camera_scan_lock:
        camera_scan_cache["checked_at"] = now_tick
        camera_scan_cache["items"] = [dict(item) for item in discovered]
    return discovered


def preview_frame_for_source(source: str, title: str) -> bytes:
    cache_key = str(source)
    now_tick = time.monotonic()
    with preview_cache_lock:
        cached = preview_frame_cache.get(cache_key)
        if cached and (now_tick - float(cached.get("created_at", 0.0))) < PREVIEW_CACHE_SECONDS:
            return cached["payload"]

    cap = open_capture(source)
    payload = placeholder_frame(title, "Camera preview unavailable")
    try:
        if cap.isOpened():
            configure_capture(cap)
            ok, frame = cap.read()
            if ok and frame is not None:
                payload = encode_jpeg(frame)
    finally:
        cap.release()

    with preview_cache_lock:
        preview_frame_cache[cache_key] = {"created_at": now_tick, "payload": payload}
    return payload


def camera_setup_slots(user_id: int, *, include_detected: bool = True) -> list[dict[str, Any]]:
    saved_rows = cameras_for_user(user_id)
    saved_by_source = {str(row["source"]): row for row in saved_rows}
    slots: list[dict[str, Any]] = []

    if include_detected:
        for detected in discover_local_camera_sources():
            source = str(detected["source"])
            saved = saved_by_source.pop(source, None)
            slots.append(
                {
                    "source": source,
                    "title": detected["label"],
                    "name": str(saved["name"]) if saved else "",
                    "camera_id": int(saved["id"]) if saved else None,
                    "connected": True,
                    "is_local": True,
                }
            )
    else:
        for offset, row in enumerate(saved_rows, start=1):
            slots.append(
                {
                    "source": str(row["source"]),
                    "title": f"Camera {offset}",
                    "name": str(row["name"]),
                    "camera_id": int(row["id"]),
                    "connected": True,
                    "is_local": str(row["source"]).isdigit(),
                }
            )
        return slots

    for offset, row in enumerate(saved_by_source.values(), start=len(slots) + 1):
        slots.append(
            {
                "source": str(row["source"]),
                "title": f"Saved camera {offset}",
                "name": str(row["name"]),
                "camera_id": int(row["id"]),
                "connected": False,
                "is_local": str(row["source"]).isdigit(),
            }
        )

    return slots


@dataclass
class CameraRuntime:
    camera_id: int
    user_id: int
    name: str
    source: str

    def __post_init__(self) -> None:
        self.state_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.pending_frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.inference_thread: threading.Thread | None = None
        self.session = None
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.pending_inference_frame: np.ndarray | None = None
        self.placeholder = placeholder_frame(self.name, "Waiting for the live feed")
        self.alert_gate = {
            "violence": {"active": False, "fingerprint": "", "last_emit": 0.0},
            "smoking": {"active": False, "fingerprint": "", "last_emit": 0.0},
            "fire": {"active": False, "fingerprint": "", "last_emit": 0.0},
        }
        self.state = base_camera_state(self.camera_id, self.name, self.source, self.user_id)
        self.last_detection = {
            "is_violent": False,
            "v_conf": 0.0,
            "is_smoking": False,
            "smoking_dets": [],
            "is_fire": False,
            "is_smoke": False,
            "fire_dets": [],
            "person_count": 0,
            "frame_skipped": False,
        }

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        if not self.inference_thread or not self.inference_thread.is_alive():
            self.inference_thread = threading.Thread(target=self.run_inference_loop, daemon=True)
            self.inference_thread.start()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                **self.state,
                "smoking_dets": [dict(item) for item in self.state["smoking_dets"]],
                "fire_dets": [dict(item) for item in self.state["fire_dets"]],
                "last_alert": dict(self.state["last_alert"]) if self.state["last_alert"] else None,
            }

    def current_jpeg(self) -> bytes:
        with self.frame_lock:
            return self.latest_jpeg or self.placeholder

    def current_frame_packet(self) -> tuple[bytes, int]:
        with self.frame_lock:
            return self.latest_jpeg or self.placeholder, self.latest_jpeg_seq

    def update_config(self, *, user_id: int, name: str, source: str) -> None:
        self.user_id = user_id
        self.name = name
        self.source = source
        self.placeholder = placeholder_frame(self.name, "Waiting for the live feed")
        with self.state_lock:
            self.state["user_id"] = user_id
            self.state["name"] = name
            self.state["source"] = source

    def mark_models_state(self) -> None:
        with self.state_lock:
            self.state["models_ready"] = shared_models.ready
            if shared_models.error:
                self.state["last_detail"] = shared_models.error

    def queue_inference(self, frame: np.ndarray) -> None:
        with self.pending_frame_lock:
            self.pending_inference_frame = frame

    def pop_pending_inference(self) -> np.ndarray | None:
        with self.pending_frame_lock:
            frame = self.pending_inference_frame
            self.pending_inference_frame = None
            return frame

    def emit_alert(self, event_type: str, active: bool, detail: str, severity: str, confidence: float | None, persons: int) -> None:
        gate = self.alert_gate[event_type]
        if not active:
            gate["active"] = False
            gate["fingerprint"] = ""
            return
        fingerprint = f"{event_type}|{detail}"
        now_tick = time.monotonic()
        should_emit = (
            not gate["active"]
            or gate["fingerprint"] != fingerprint
            or now_tick - gate["last_emit"] >= {"violence": 5.0, "smoking": 4.0, "fire": 5.0}[event_type]
        )
        gate["active"] = True
        gate["fingerprint"] = fingerprint
        if not should_emit:
            return
        gate["last_emit"] = now_tick
        alert = record_alert(
            user_id=self.user_id,
            camera_id=self.camera_id,
            camera_name=self.name,
            event_type=event_type,
            severity=severity,
            detail=detail,
            confidence=confidence,
            persons=persons,
        )
        with self.state_lock:
            self.state["last_alert"] = alert
            self.state["last_alert_id"] = alert["id"]

    def apply_inference_result(self, result: dict[str, Any]) -> None:
        with self.result_lock:
            last = dict(self.last_detection)
            last["frame_skipped"] = result.get("skipped", False)
            if not last["frame_skipped"]:
                persons = int(result.get("person_count", 0))
                violent = bool(result.get("is_violent", False))
                last["is_violent"] = violent
                last["v_conf"] = float(result.get("violence_confidence", 0.0)) if violent else 0.0
                last["is_smoking"] = bool(result.get("is_smoking", False))
                last["smoking_dets"] = [
                    {"bbox": list(item["bbox"]), "conf": float(item["confidence"]), "label": str(item["label"])}
                    for item in result.get("smoking_detections", [])
                ]
                last["is_fire"] = bool(result.get("is_fire", False))
                last["is_smoke"] = bool(result.get("is_smoke", False))
                last["fire_dets"] = [
                    {"bbox": list(item["bbox"]), "conf": float(item["confidence"]), "label": str(item["label"])}
                    for item in result.get("fire_detections", [])
                ]
                last["person_count"] = persons
            self.last_detection = last

        smoking_count = len(last["smoking_dets"])
        fire_active = bool(last["is_fire"] or last["is_smoke"])
        v_pct = int(round(last["v_conf"] * 100))
        fire_label = (
            "Fire and smoke"
            if last["is_fire"] and last["is_smoke"]
            else "Fire"
            if last["is_fire"]
            else "Smoke"
            if last["is_smoke"]
            else "Clear"
        )
        self.emit_alert("violence", last["is_violent"], f"Confidence {v_pct}%", "HIGH", last["v_conf"], int(last["person_count"]))
        self.emit_alert("smoking", smoking_count > 0, f"{smoking_count} detection{'s' if smoking_count != 1 else ''}", "MEDIUM", float(smoking_count), int(last["person_count"]))
        self.emit_alert("fire", fire_active, fire_label, "HIGH", 1.0 if fire_active else 0.0, int(last["person_count"]))

    def run_inference_loop(self) -> None:
        inference_idle_sleep = min(ACTIVE_SLEEP, 0.01)
        while not self.stop_event.is_set():
            if not shared_models.ready:
                time.sleep(inference_idle_sleep)
                continue

            if self.session is None and HAS_API:
                self.session = InferenceSession(
                    shared_models.v_model,
                    shared_models.l_model,
                    shared_models.f_model,
                    shared_models.p_model,
                )
            if self.session is None:
                time.sleep(inference_idle_sleep)
                continue

            frame = self.pop_pending_inference()
            if frame is None:
                time.sleep(inference_idle_sleep)
                continue

            with shared_models.inference_lock:
                result = self.session.infer(frame)
            self.apply_inference_result(result)

    def run(self) -> None:
        shared_models.ensure_loaded()
        fps_cap = max(int(app_module.FPS_CAP or 0), TARGET_CAMERA_FPS)
        frame_interval = (1.0 / fps_cap) if fps_cap > 0 else 0.0
        inference_interval = (1.0 / MAX_INFERENCE_FPS) if MAX_INFERENCE_FPS > 0 else 0.0
        prev_gray = None
        frame_count = 0
        fps_counter = 0
        fps_display = 0.0
        t_fps = time.time()
        t_last = 0.0
        last_inference_request_at = 0.0
        started_at = time.monotonic()

        while not self.stop_event.is_set():
            self.mark_models_state()
            if not shared_models.ready:
                with self.state_lock:
                    self.state["camera_online"] = False
                    self.state["updated_at"] = now_iso()
                    self.state["uptime_sec"] = int(time.monotonic() - started_at)
                time.sleep(0.5)
                continue

            cap = open_capture(self.source)
            if not cap.isOpened():
                with self.state_lock:
                    self.state["camera_online"] = False
                    self.state["updated_at"] = now_iso()
                    self.state["last_detail"] = "Camera not available"
                time.sleep(2.0)
                continue

            configure_capture(cap)

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    with self.state_lock:
                        self.state["camera_online"] = False
                        self.state["updated_at"] = now_iso()
                        self.state["last_detail"] = "Connection lost"
                    break

                if frame_interval > 0:
                    now_tick = time.time()
                    wait = frame_interval - (now_tick - t_last)
                    if wait > 0:
                        time.sleep(wait)
                t_last = time.time()
                stamp = now_local()

                frame_count += 1
                fps_counter += 1
                if fps_counter >= 15:
                    elapsed = max(time.time() - t_fps, 1e-6)
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    t_fps = time.time()

                curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
                motion = prev_gray is None or app_module.has_motion(prev_gray, curr_gray)
                prev_gray = curr_gray

                payload = encode_jpeg(frame)
                if payload:
                    with self.frame_lock:
                        self.latest_jpeg = payload
                        self.latest_jpeg_seq += 1

                if motion and self.session is not None:
                    now_monotonic = time.monotonic()
                    infer_due = inference_interval <= 0 or (now_monotonic - last_inference_request_at) >= inference_interval
                    if infer_due:
                        self.queue_inference(frame.copy())
                        last_inference_request_at = now_monotonic

                with self.result_lock:
                    last = dict(self.last_detection)

                smoking_count = len(last["smoking_dets"])
                fire_active = bool(last["is_fire"] or last["is_smoke"])
                alert_type = ""
                if last["is_violent"]:
                    alert_type = "violence"
                elif fire_active:
                    alert_type = "fire"
                elif smoking_count > 0:
                    alert_type = "smoking"

                v_pct = int(round(last["v_conf"] * 100))
                fire_label = "Fire and smoke" if last["is_fire"] and last["is_smoke"] else "Fire" if last["is_fire"] else "Smoke" if last["is_smoke"] else "Clear"

                with self.state_lock:
                    self.state.update(last)
                    self.state["camera_online"] = True
                    self.state["models_ready"] = shared_models.ready
                    self.state["fps"] = round(fps_display, 1)
                    self.state["frame"] = frame_count
                    self.state["video_timestamp"] = stamp.strftime("%Y-%m-%d %H:%M:%S")
                    self.state["updated_at"] = stamp.isoformat(timespec="seconds")
                    self.state["uptime_sec"] = int(time.monotonic() - started_at)
                    self.state["active_alert"] = bool(alert_type)
                    self.state["alert_type"] = alert_type
                    self.state["last_detail"] = fire_label if fire_active else f"{smoking_count} detection{'s' if smoking_count != 1 else ''}" if smoking_count else f"{v_pct}% confidence" if last["is_violent"] else "Monitoring"
            cap.release()


def refresh_camera_runtimes() -> None:
    desired_rows = fetch_all("SELECT * FROM cameras WHERE enabled = 1 ORDER BY sort_order, id")
    desired = {int(row["id"]): row for row in desired_rows}
    with registry_lock:
        for camera_id, runtime in list(camera_runtimes.items()):
            row = desired.get(camera_id)
            if row is None:
                runtime.stop()
                del camera_runtimes[camera_id]
                continue
            if runtime.name != row["name"] or runtime.source != row["source"] or runtime.user_id != row["user_id"]:
                runtime.stop()
                del camera_runtimes[camera_id]
        for camera_id, row in desired.items():
            if camera_id in camera_runtimes:
                continue
            runtime = CameraRuntime(
                camera_id=camera_id,
                user_id=int(row["user_id"]),
                name=str(row["name"]),
                source=str(row["source"]),
            )
            camera_runtimes[camera_id] = runtime
            runtime.start()


def pick_focus_camera(states: list[dict[str, Any]]) -> int | None:
    if not states:
        return None
    priorities = {"violence": 3, "fire": 2, "smoking": 1, "": 0}
    best = max(states, key=lambda item: (priorities.get(item.get("alert_type", ""), 0), item.get("updated_at", "")))
    return int(best["camera_id"]) if best.get("active_alert") else int(states[0]["camera_id"])


@flask_app.before_request
def load_user() -> None:
    g.user = get_user_by_id(session.get("user_id"))


@flask_app.context_processor
def shared_context() -> dict[str, Any]:
    return {
        "current_user": g.get("user"),
        "current_year": now_local().year,
    }


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not g.user:
            next_url = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            return redirect(url_for("login", next=next_url))
        return view(*args, **kwargs)

    return wrapped


def setup_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user and not user_has_cameras(int(g.user["id"])) and request.endpoint not in {"setup", "logout"}:
            return redirect(url_for("setup"))
        return view(*args, **kwargs)

    return wrapped


def current_camera_states(user_id: int) -> list[dict[str, Any]]:
    rows = cameras_for_user(user_id)
    states: list[dict[str, Any]] = []
    with registry_lock:
        for row in rows:
            runtime = camera_runtimes.get(int(row["id"]))
            if runtime:
                states.append(runtime.snapshot())
            else:
                states.append(base_camera_state(int(row["id"]), str(row["name"]), str(row["source"]), int(row["user_id"])))
    return states


def selected_camera_id_for_request(user_id: int) -> int | None:
    requested = request.args.get("camera", type=int)
    states = current_camera_states(user_id)
    if requested and any(state["camera_id"] == requested for state in states):
        return requested
    return pick_focus_camera(states)


def read_camera_form(prefix_name: str = "camera_name", prefix_source: str = "camera_source") -> list[dict[str, str]]:
    names = request.form.getlist(prefix_name)
    sources = request.form.getlist(prefix_source)
    cameras: list[dict[str, str]] = []
    for index, name in enumerate(names):
        source = sources[index] if index < len(sources) else str(index)
        if not name.strip():
            continue
        cameras.append({"name": name.strip(), "source": source.strip() or str(index)})
    return cameras


@flask_app.after_request
def no_store(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@flask_app.route("/")
def landing():
    if g.user:
        return redirect(url_for("dashboard" if user_has_cameras(int(g.user["id"])) else "setup"))
    return render_template("landing.html", page_title="Plithos", active_page="landing")


@flask_app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard" if user_has_cameras(int(g.user["id"])) else "setup"))
    first_user = count_users() == 0
    next_url = request.values.get("next", "").strip() or request.args.get("next", "").strip()
    if request.method == "POST":
        if first_user:
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")
            if not full_name:
                flash("Enter your name to create the first account.", "error")
            elif not validate_email(email):
                flash("Enter a valid email address.", "error")
            elif len(password) < 8:
                flash("Use at least 8 characters for the password.", "error")
            elif password != confirm:
                flash("Passwords do not match.", "error")
            else:
                user_id = create_user(full_name, email, password)
                session["user_id"] = user_id
                refresh_camera_runtimes()
                return redirect(url_for("setup"))
        else:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = get_user_by_email(email)
            if not user or not check_password_hash(user["password_hash"], password):
                flash("Incorrect email or password.", "error")
            else:
                session["user_id"] = int(user["id"])
                return redirect(next_url or url_for("dashboard" if user_has_cameras(int(user["id"])) else "setup"))
    return render_template("login.html", page_title="Login", active_page="login", first_user=first_user, next_url=next_url)


@flask_app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("landing"))


@flask_app.route("/setup", methods=["GET", "POST"])
@login_required
def setup():
    user_id = int(g.user["id"])
    if request.method == "POST":
        cameras = read_camera_form()
        if not cameras:
            flash("Name at least one camera to continue.", "error")
        else:
            replace_cameras(user_id, cameras)
            refresh_camera_runtimes()
            flash("Cameras saved.", "success")
            return redirect(url_for("dashboard"))
    slots = camera_setup_slots(user_id)
    return render_template(
        "setup.html",
        page_title="Setup",
        active_page="setup",
        slots=slots,
    )


@flask_app.route("/dashboard")
@login_required
@setup_required
def dashboard():
    cameras = cameras_for_user(int(g.user["id"]))
    selected_camera_id = selected_camera_id_for_request(int(g.user["id"]))
    return render_template(
        "dashboard.html",
        page_title="Dashboard",
        active_page="dashboard",
        cameras=cameras,
        selected_camera_id=selected_camera_id,
        preferences={
            "theme": g.user["theme"],
            "sound_enabled": bool(g.user["sound_enabled"]),
            "auto_switch_alerts": bool(g.user["auto_switch_alerts"]),
        },
    )


@flask_app.route("/cameras")
@login_required
@setup_required
def cameras_page():
    cameras = cameras_for_user(int(g.user["id"]))
    return render_template(
        "cameras.html",
        page_title="All Cameras",
        active_page="cameras",
        cameras=cameras,
        preferences={
            "theme": g.user["theme"],
            "sound_enabled": bool(g.user["sound_enabled"]),
        },
    )


@flask_app.route("/logs")
@login_required
@setup_required
def logs():
    user_id = int(g.user["id"])
    camera_id = request.args.get("camera", type=int)
    event_type = request.args.get("type", "all")
    search = request.args.get("search", "").strip()
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    rows = alerts_for_user(
        user_id,
        limit=MAX_LOG_ROWS,
        camera_id=camera_id,
        event_type=event_type,
        search=search,
        date_from=date_from,
        date_to=date_to,
    )
    by_type = {"violence": 0, "smoking": 0, "fire": 0}
    by_day: dict[str, int] = {}
    for row in rows:
        by_type[row["event_type"]] = by_type.get(row["event_type"], 0) + 1
        day_key = row["created_at"][:10]
        by_day[day_key] = by_day.get(day_key, 0) + 1
    trend = [{"label": key, "count": by_day[key]} for key in sorted(by_day.keys())[-7:]]
    type_max = max(by_type.values()) if by_type else 0
    trend_max = max((item["count"] for item in trend), default=0)
    return render_template(
        "logs.html",
        page_title="Logs",
        active_page="logs",
        cameras=cameras_for_user(user_id),
        rows=rows,
        filters={
            "camera": camera_id,
            "type": event_type,
            "search": search,
            "date_from": date_from,
            "date_to": date_to,
        },
        type_summary=by_type,
        trend=trend,
        type_max=type_max,
        trend_max=trend_max,
    )


@flask_app.route("/settings", methods=["GET", "POST"])
@login_required
@setup_required
def settings():
    user_id = int(g.user["id"])
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            existing = get_user_by_email(email) if email else None
            if not full_name:
                flash("Enter a name for the account.", "error")
            elif not validate_email(email):
                flash("Enter a valid email address.", "error")
            elif existing and int(existing["id"]) != user_id:
                flash("That email is already in use.", "error")
            else:
                payload: dict[str, Any] = {"full_name": full_name, "email": email}
                if not g.user["report_email"] or str(g.user["report_email"]).strip().lower() == str(g.user["email"]).strip().lower():
                    payload["report_email"] = email
                update_user_settings(user_id, payload)
                flash("Account details updated.", "success")
        elif action == "preferences":
            update_user_settings(
                user_id,
                {
                    "theme": request.form.get("theme", "dark"),
                    "sound_enabled": 1 if request.form.get("sound_enabled") else 0,
                    "auto_switch_alerts": 1 if request.form.get("auto_switch_alerts") else 0,
                    "notifications_enabled": 1 if request.form.get("notifications_enabled") else 0,
                },
            )
            flash("Preferences saved.", "success")
        elif action == "alerts":
            report_email = request.form.get("report_email", "").strip().lower()
            subject = request.form.get("alert_subject", "").strip()
            message = request.form.get("alert_message", "").strip()
            if not validate_email(report_email):
                flash("Enter a valid email address for reports.", "error")
            else:
                update_user_settings(
                    user_id,
                    {
                        "report_email": report_email,
                        "alert_subject": subject or default_alert_subject(),
                        "alert_message": message or default_alert_message(),
                    },
                )
                flash("Alert email settings updated.", "success")
        elif action == "password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")
            if not check_password_hash(str(g.user["password_hash"]), current_password):
                flash("Enter your current password to make a change.", "error")
            elif len(new_password) < 8:
                flash("Use at least 8 characters for the new password.", "error")
            elif new_password != confirm_password:
                flash("The new passwords do not match.", "error")
            else:
                update_user_settings(user_id, {"password_hash": generate_password_hash(new_password)})
                flash("Password updated.", "success")
        elif action == "cameras":
            cameras = read_camera_form()
            if not cameras:
                flash("Name at least one camera to keep monitoring.", "error")
            else:
                replace_cameras(user_id, cameras)
                refresh_camera_runtimes()
                flash("Camera settings updated.", "success")
        elif action == "send_test_email":
            subject = str(g.user["alert_subject"] or default_alert_subject()).strip()
            message = "\n".join(
                [
                    str(g.user["alert_message"] or default_alert_message()).strip(),
                    "",
                    "This is a test email from Plithos.",
                    f"Time: {human_time(now_iso())}",
                ]
            )
            ok, error = send_email_message(
                to_address=report_recipient(g.user),
                subject=subject,
                body=message,
                user=g.user,
            )
            if ok:
                flash("Test email sent.", "success")
            else:
                flash(error or "Email delivery is not ready yet.", "error")
        return redirect(url_for("settings"))

    return render_template(
        "settings.html",
        page_title="Settings",
        active_page="settings",
        cameras=cameras_for_user(user_id),
        camera_slots=camera_setup_slots(user_id, include_detected=False),
    )


@flask_app.route("/api/state")
@login_required
@setup_required
def api_state():
    user_id = int(g.user["id"])
    states = current_camera_states(user_id)
    recent_alerts = [alert_row_to_dict(row) for row in alerts_for_user(user_id, limit=MAX_RECENT_ALERTS)]
    return jsonify(
        {
            "generated_at": now_iso(),
            "models_ready": shared_models.ready,
            "model_error": shared_models.error,
            "settings": {
                "theme": g.user["theme"],
                "sound_enabled": bool(g.user["sound_enabled"]),
                "auto_switch_alerts": bool(g.user["auto_switch_alerts"]),
                "notifications_enabled": bool(g.user["notifications_enabled"]),
            },
            "cameras": states,
            "recent_alerts": recent_alerts,
            "alerts_total": total_alerts_for_user(user_id),
            "active_alert_camera_id": pick_focus_camera(states),
        }
    )


def stream_camera(camera_id: int):
    placeholder = placeholder_frame("Plithos", "Camera not configured")
    last_seq = -1
    while True:
        with registry_lock:
            runtime = camera_runtimes.get(camera_id)
        if runtime:
            payload, seq = runtime.current_frame_packet()
        else:
            payload, seq = placeholder, -1
        if seq != last_seq or not runtime:
            last_seq = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        time.sleep(ACTIVE_SLEEP if runtime else IDLE_SLEEP)


def snapshot_payload(camera_id: int) -> bytes:
    placeholder = placeholder_frame("Plithos", "Camera not configured")
    with registry_lock:
        runtime = camera_runtimes.get(camera_id)
    return runtime.current_jpeg() if runtime else placeholder


@flask_app.route("/video_feed/<int:camera_id>")
@login_required
@setup_required
def video_feed(camera_id: int):
    if not any(int(row["id"]) == camera_id for row in cameras_for_user(int(g.user["id"]))):
        abort(404)
    return Response(
        stream_camera(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@flask_app.route("/snapshot/<int:camera_id>")
@login_required
@setup_required
def snapshot_feed(camera_id: int):
    if not any(int(row["id"]) == camera_id for row in cameras_for_user(int(g.user["id"]))):
        abort(404)
    return Response(snapshot_payload(camera_id), mimetype="image/jpeg")


@flask_app.route("/setup_preview/<int:source_index>")
@login_required
def setup_preview(source_index: int):
    allowed_sources = {int(item["source"]) for item in discover_local_camera_sources() if str(item["source"]).isdigit()}
    if source_index not in allowed_sources:
        abort(404)
    return Response(preview_frame_for_source(str(source_index), f"Camera {source_index + 1}"), mimetype="image/jpeg")


@flask_app.route("/monitor")
def monitor_redirect():
    return redirect(url_for("dashboard"))


@flask_app.route("/analytics")
def analytics_redirect():
    return redirect(url_for("logs"))


@flask_app.route("/alert-triage")
def alert_redirect():
    return redirect(url_for("logs"))


@flask_app.route("/health")
def health():
    return jsonify({"ok": True, "models_ready": shared_models.ready, "time": now_iso()})


def build_readme_school_copy() -> str:
    return "\n".join(
        [
            "# Plithos",
            "",
            "Smart safety monitoring for schools.",
            "",
            "Plithos is an AI-powered monitoring system built for school environments. It watches live camera feeds and helps staff respond faster by detecting fights, fire hazards, and smoking in real time.",
            "",
            "## What It Does",
            "",
            "- Detects fights and aggressive behavior",
            "- Detects fire and smoke hazards",
            "- Tracks smoking incidents",
            "- Shows live camera feeds and alerts in a web dashboard",
            "- Sends alert emails with a CSV report",
            "",
            "## Main Pages",
            "",
            "- Landing page",
            "- Login",
            "- Camera setup",
            "- Dashboard",
            "- All cameras",
            "- Logs",
            "- Settings",
            "",
            "## Technology",
            "",
            "- Python",
            "- Flask",
            "- YOLO",
            "- OpenCV",
            "- AWS",
            "",
            "## Privacy",
            "",
            "Plithos focuses on situations, not identity. It does not use facial recognition.",
            "",
            "## Team",
            "",
            "- Mohammed Fardan â€” Team Lead / Cloud / Frontend",
            "- Yousif Alaali â€” Cloud / Database",
            "- Ali Yasser â€” Software Developer / AI Integration",
            "- Salman Ashoor â€” Software Developer / Hardware / R&D",
            "",
            f"Â© {now_local().year} Plithos. All rights reserved.",
        ]
    )


def init_application() -> None:
    init_db()
    shared_models.ensure_loaded()
    refresh_camera_runtimes()


init_application()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plithos Web Server")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"\n[Plithos] Web interface -> http://localhost:{args.port}\n")
    flask_app.run(host=args.host, port=args.port, threaded=True, use_reloader=False, debug=False)

