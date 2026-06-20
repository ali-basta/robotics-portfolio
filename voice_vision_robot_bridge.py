# === main_mcp.py ===
import os
import sys
import json
import subprocess
import tempfile
import wave
import requests
import re
import sounddevice as sd
import numpy as np
import queue
from vosk import Model, KaldiRecognizer
import webrtcvad
import cv2, threading, time, platform, traceback, base64
import math
from openai import OpenAI  # new SDK
oai = OpenAI()             # reads OPENAI_API_KEY from env

USE_LLM = os.getenv("LLM_PROVIDER","openai").lower()  # 'openai' or 'ollama'
if USE_LLM == "ollama":
    import ollama_client as oll

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


def point_inside_box(px, py, box, margin=5):
    x1, y1, x2, y2 = box["bbox"]
    return (x1 - margin) <= px <= (x2 + margin) and \
           (y1 - margin) <= py <= (y2 + margin)


def filter_coins(coin_detections, boxes, ref_markers, frame, tf):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    filtered = []

    def inside_any_box(px, py, margin=10):
        for b in boxes.values():
            x1, y1, x2, y2 = b["bbox"]
            if (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin):
                return True
        return False

    def near_reference(px, py, ref, dist=60):
        for name, (rx, ry) in ref.items():
            if (px - rx)**2 + (py - ry)**2 < dist*dist:
                return True
        return False

    for d in coin_detections:
        cx = int(d["cx"])
        cy = int(d["cy"])

        # reject inside any box
        if inside_any_box(cx, cy):
            continue

        # reject near reference markers
        if near_reference(cx, cy, ref_markers):
            continue

        # reject low saturation (dull objects)
        if hsv[cy, cx][1] < 35:
            continue

        # convert to plate coordinates
        x_mm, y_mm = pixels_to_plate(cx, cy, tf)

        # reject outside plate
        if not (0 <= x_mm <= 340 and 0 <= y_mm <= 340):
            continue

        # reject coins beyond X = 215 mm
        if x_mm > 215:
            continue

        d["x_mm"] = x_mm
        d["y_mm"] = y_mm
        filtered.append(d)

    return filtered



# ================== Circle + Color Detection (coins) ==================
def detect_circles(frame):
    """
    Stricter circle detection tuned for ZV-1A top view.
    Reduces false positives while keeping real coins visible.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast but avoid exaggerating noise
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # Edge map
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Stricter Hough parameters
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.4,
        minDist=80,
        param1=180,
        param2=50,
        minRadius=18,
        maxRadius=40
    )

    if circles is not None:
        return np.uint16(np.around(circles[0, :]))
    return []

def get_circle_color(frame, x, y, r):
    """Average color in the circle area with extra blur for stable HSV."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    mean_bgr = cv2.mean(blurred, mask=mask)[:3]
    return tuple(map(int, mean_bgr))


def classify_color(bgr):
    """
    HSV classification tuned for Sony ZV-1A.
    Detects red, orange, green, blue (including dark blue) — stable, no purple.
    """
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # --- red ---
    # true red: near 0° or 180°, high saturation
    if (h <= 6 or h >= 170) and s >= 30 and v >= 25:
        return "red"
    if 17 <= h <= 21 and s >= 60 and v < 80:
        return "red"     # deeper reds with slight hue drift

    # --- orange ---
    # lower-saturation or slightly brighter region between red and yellow
    if 7 <= h <= 17 and s >= 60 and v >= 90:
        return "orange"

    # --- green ---
    if 35 <= h <= 95 and s >= 25 and v >= 45:
        return "green"

    # --- blue (bright + dark navy) ---
    if 85 <= h <= 140 and 30 <= s <= 255 and 40 <= v <= 255:
        return "blue"

    return "unknown"




# === CRITICAL: restore old working coin coordinate convention ===
def bbox_center_mm_coin(bbox, roi_w, roi_h):
    """
    Old working convention from your original program.
    Returns (x_mm, y_mm) as (old_y_mm, old_x_mm).
    """
    x1, y1, x2, y2 = bbox
    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x_from_left_px = roi_w - xc
    old_x_mm = x_from_left_px / PIXELS_PER_MM
    old_y_mm = yc / PIXELS_PER_MM
    return old_y_mm, old_x_mm, xc, yc  # note order



def detect_on_frame(frame):
    roi = frame
    circles = detect_circles(roi)
    boxes = detect_boxes_by_color(frame.copy())   # <-- added

    detections = []
    for det_id, (cx, cy, r) in enumerate(circles):
        col_bgr = get_circle_color(roi, cx, cy, r)
        color_name = classify_color(col_bgr)
        if color_name == "unknown":
            continue

        # ignore orange reference markers
        if color_name == "orange" and r > 30:
            continue

        # NEW: skip coins inside any box
        skip = False
        for b in boxes.values():
            if point_inside_box(cx, cy, b):
                skip = True
                break
        if skip:
            continue

        bbox = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
        detections.append({
            "id": det_id,
            "bbox": bbox,
            "cx": float(cx),
            "cy": float(cy),
            "radius": int(r),
            "area_px": int(math.pi * r * r),
            "color": color_name
        })

    return detections



def _annotate_colors_on_frame(frame):
    """
    Draw ONLY coins and boxes inside the 340x340 mm plate.
    Everything outside is ignored.
    """
    if frame is None:
        return None

    roi = frame

    # Detect reference markers
    ref = detect_reference_markers(frame.copy())
    tf = compute_plate_transform(ref) if ref else None

    # Detect boxes + circles
    boxes_px = detect_boxes_by_color(frame.copy())
    circles = detect_circles(roi)

    # Helpers
    def inside_plate_px(u, v):
        if tf is None:
            return False
        inside, _, _ = inside_plate_mm(u, v, tf)
        return inside

    def point_inside_box(x, y, box):
        x1, y1, x2, y2 = box["bbox"]
        return x1 <= x <= x2 and y1 <= y <= y2

    # ----------------------------------------------------------
    # DRAW BOXES (ONLY inside plate, NO white center dot)
    # ----------------------------------------------------------
    for color, info in boxes_px.items():
        xc = info["xc"]
        yc = info["yc"]

        if not inside_plate_px(xc, yc):
            continue  # skip boxes outside plate

        x1, y1, x2, y2 = info["bbox"]

        # outline color
        if color == "red":
            col = (0, 0, 255)
        elif color == "green":
            col = (0, 255, 0)
        elif color == "blue":
            col = (255, 0, 0)
        else:
            col = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)

        cv2.putText(frame, f"{color} box", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 3)

    # ----------------------------------------------------------
    # DRAW COINS (ONLY inside plate AND outside boxes)
    # ----------------------------------------------------------
    for (cx, cy, r) in circles:

        # 1) skip if outside plate
        if not inside_plate_px(cx, cy):
            continue

        # 2) skip if inside any box (collision avoidance)
        in_box = False
        for b in boxes_px.values():
            if point_inside_box(cx, cy, b):
                in_box = True
                break
        if in_box:
            continue

        # 3) detect color
        col_bgr = get_circle_color(roi, cx, cy, r)
        color_name = classify_color(col_bgr)

        # keep orange now
        if color_name == "unknown":
            continue

        # draw the coin
        cv2.circle(frame, (int(cx), int(cy)), r, (0, 255, 0), 3)

        cv2.putText(frame, color_name,
                    (int(cx - r), int(cy - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 3)

    return frame



# ================== BOX DETECTION (treat as colored blobs, full frame) ==================
BOX_COLORS = ["red","green","blue"]

def color_mask_ranges():
    return {
        "red":  [((0,  40, 50), (10, 255, 255)),
                 ((170,40, 50), (180,255,255))],
        "green":[((30, 30, 40), (90, 255, 255))],
        "blue": [((85, 30, 40), (140,255,255))],
    }

def mask_for_color(hsv_img, color):
    ranges = color_mask_ranges()[color]
    msum = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv_img, np.array(lo, np.uint8), np.array(hi, np.uint8))
        msum = m if msum is None else cv2.bitwise_or(msum, m)
    msum = cv2.morphologyEx(msum, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    msum = cv2.morphologyEx(msum, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    return msum

def bbox_center_mm_box_full(bbox, full_w, full_h):
    """
    Use the exact same convention as coins but on the FULL frame.
    Returns (x_boxmm, y_boxmm) as (old_y_mm, old_x_mm).
    """
    x1, y1, x2, y2 = bbox
    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x_from_left_px = full_w - xc
    old_x_mm = x_from_left_px / PIXELS_PER_MM
    old_y_mm = yc / PIXELS_PER_MM
    return old_y_mm, old_x_mm, xc, yc



BOX_COLORS = ["red", "green", "blue"]

def detect_boxes_by_color(frame):
    """
    Detect one large blob per box color over the FULL frame.
    Returns dict: color -> {"bbox":(x1,y1,x2,y2), "xc":..., "yc":...}
    """
    roi = cv2.GaussianBlur(frame.copy(), (5,5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    out = {}
    for color in BOX_COLORS:
        mask = mask_for_color(hsv, color)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 800:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, x+w, y+h)
        xc = x + w / 2.0
        yc = y + h / 2.0
        out[color] = {"bbox": bbox, "xc": float(xc), "yc": float(yc)}
    return out




# ================== REFERENCE MARKERS (orange corners) ==================

# bottom marker position relative to top marker (origin)
# world axes: X -> right, Y -> down
PLATE_DX_MM = 340.0   # bottom is 340 mm to the LEFT of the top marker
PLATE_DY_MM =  340.0   # bottom is 340 mm DOWN from the top marker
PLATE_WIDTH_MM  = 340.0
PLATE_HEIGHT_MM = 340.0

def detect_reference_markers(frame):
    """
    Detect the two orange corner rectangles on the FULL frame.
    Returns dict: {"top": (u_top, v_top), "bottom": (u_bot, v_bot)} in pixels,
    or None if not found reliably.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # orange range – can be tuned if needed
    lower_orange = np.array([5, 150, 150], np.uint8)
    upper_orange = np.array([25, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) < 2:
        return None

    # take the two largest orange blobs
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    centers = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy))

    if len(centers) != 2:
        return None

    # sort by y: smaller y = top, larger y = bottom
    centers.sort(key=lambda p: p[1])
    top = centers[0]
    bottom = centers[1]
    return {"top": top, "bottom": bottom}


def compute_plate_transform(ref_points):
    """
    From two reference points (pixel) and their known plate positions (mm),
    compute a similarity transform (rotation + scale + translation).
    Returns dict with parameters or None if invalid.
    """
    (uA, vA) = ref_points["top"]
    (uB, vB) = ref_points["bottom"]

    # pixel vector from top to bottom
    dx_img = uB - uA
    dy_img = vB - vA
    dist_img = math.hypot(dx_img, dy_img)
    if dist_img < 1e-6:
        return None

    # plate vector from top to bottom (mm)
    dx_plate = PLATE_DX_MM
    dy_plate = PLATE_DY_MM
    dist_plate = math.hypot(dx_plate, dy_plate)

    scale = dist_plate / dist_img

    angle_img = math.atan2(dy_img, dx_img)
    angle_plate = math.atan2(dy_plate, dx_plate)
    rotation = angle_plate - angle_img

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    return {
        "u0": uA,
        "v0": vA,
        "scale": scale,
        "cos": cos_r,
        "sin": sin_r,
    }


def pixels_to_plate(u, v, tf):
    """
    Convert pixel coordinates (u, v) to plate mm coordinates (X, Y),
    using the transform returned by compute_plate_transform().
    Origin is at the TOP orange marker center.
    """
    du = u - tf["u0"]
    dv = v - tf["v0"]

    du_rot = du * tf["cos"] - dv * tf["sin"]
    dv_rot = du * tf["sin"] + dv * tf["cos"]

    X = du_rot * tf["scale"]
    Y = dv_rot * tf["scale"]
    return X, Y


def inside_plate_mm(u, v, tf, margin_mm=0.0):
    """
    Check if pixel (u,v) lies inside the 340x340 mm plate.
    Returns (inside_bool, X_mm, Y_mm).
    """
    X, Y = pixels_to_plate(u, v, tf)
    inside = (
        -margin_mm <= X <= PLATE_WIDTH_MM + margin_mm and
        -margin_mm <= Y <= PLATE_HEIGHT_MM + margin_mm
    )
    return inside, X, Y



def _annotate_reference_markers_on_frame(frame):
    ref = detect_reference_markers(frame.copy())
    if not ref:
        return frame
    for label, (u, v) in ref.items():
        cv2.circle(frame, (int(u), int(v)), 8, (0, 165, 255), 2)  # orange circle
        cv2.putText(
            frame,
            label,
            (int(u) + 10, int(v) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame




def _annotate_boxes_on_frame(frame):
    boxes = detect_boxes_by_color(frame)
    bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
    for color, info in boxes.items():
        x1, y1, x2, y2 = info["bbox"]
        #cv2.rectangle(frame, (x1, y1), (x2, y2), bgr.get(color, (255, 255, 255)), 3)
        label = f"{color} box"
        # larger, bold text for HD
        #cv2.putText(
            #frame,
           # label,
            #(x1 + 5, max(20, y1 - 10)),
           # cv2.FONT_HERSHEY_SIMPLEX,
           # 1.2,                 # font size up
            #(255, 255, 255),
           # 3,                   # thicker outline
           # cv2.LINE_AA
       # )
        cv2.circle(frame, (int(info["xc"]), int(info["yc"])), 6, (255, 255, 255), -1)
    return frame

# ================== BATCH SENDER ==================
def send_coords_batch(items: list[dict]) -> None:
    payload = {
        "coords": [
            {
                "x_mm": round(float(it["x_mm"]), 3),
                "y_mm": round(float(it["y_mm"]), 3),
                "color": str(it.get("color","")).lower().strip(),
                "box":   str(it.get("box","")).lower().strip(),
                **({"x_boxmm": round(float(it["x_boxmm"]), 3)} if it.get("x_boxmm") is not None else {}),
                **({"y_boxmm": round(float(it["y_boxmm"]), 3)} if it.get("y_boxmm") is not None else {})
            }
            for it in items
        ]
    }
    try:
        r = requests.post(DEST_URL, json=payload, timeout=5)
        print(f"[OK] Sent {len(payload['coords'])} coords -> {r.status_code}")
    except Exception as e:
        print(f"[ERR] Failed to send batch coords: {e}")

def apply_masks(hsv_img, hsv_ranges):
    mask_total = None
    for r in hsv_ranges:
        mask = cv2.inRange(hsv_img, np.array(r["lower"],np.uint8), np.array(r["upper"],np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)
    return mask_total

def select_detections(detections, amount, order):
    if order not in VALID_ORDERS: order="left_to_right"
    if order=="right_to_left": detections=sorted(detections,key=lambda d:d["bbox"][0], reverse=True)
    elif order=="left_to_right": detections=sorted(detections,key=lambda d:d["bbox"][0])
    elif order=="bottom_to_top": detections=sorted(detections,key=lambda d:d["bbox"][1], reverse=True)
    elif order=="top_to_bottom": detections=sorted(detections,key=lambda d:d["bbox"][1])
    elif order=="largest_first": detections=sorted(detections,key=lambda d:d["area_px"],reverse=True)
    elif order=="smallest_first": detections=sorted(detections,key=lambda d:d["area_px"])
    k=len(detections) if str(amount)=="all" else int(amount)
    return detections[:max(0,min(k,len(detections)))]

# ================== CONFIG ==================
SERVER_HOST = os.getenv("LOCAL_SERVER_HOST", "<server-ip>")
DEST_URL = f"http://{SERVER_HOST}:8000/coords"   # robot server
CROP = (int(300 * 3.0), int(0 * 2.25), int(530 * 3.0), int(480 * 2.25))
PIXELS_PER_MM = 1.329 * 3.0  * 0.748  # ≈ 3.987
MIN_RADIUS = 10
MAX_RADIUS = 35
MIN_DIST   = 30
MIN_AREA   = 200
COORDS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coords.json")
# NOTE: legacy/unused — leftover from an earlier MCP-based iteration of this project.
# Not called anywhere in the current pipeline; kept for reference.
COLOR_PROCESSOR_URL = f"http://{SERVER_HOST}:6000/color"
TARGET_W, TARGET_H = 1920, 1080
CAM_NAME  = os.getenv("CAM_NAME", "").strip()
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_FPS   = int(os.getenv("CAM_FPS", "60"))
VALID_ORDERS = {"left_to_right","right_to_left","top_to_bottom","bottom_to_top","largest_first","smallest_first"}

# ================== Camera helpers ==================
import platform, cv2, time, threading, traceback

def _safe_set(cap, prop, value):
    try:
        cap.set(prop, value)
    except Exception as e:
        print(f"[Camera] WARN cap.set({prop},{value}): {e}", flush=True)

def _test_read(cap):
    try:
        ok, frame = cap.read()
        return bool(ok and frame is not None), frame
    except Exception as e:
        print(f"[Camera] WARN cap.read(): {e}", flush=True)
        return False, None

def _apply_common_caps(cap, width, height, fps, force_mjpg=True):
    """
    Modified to support Sony Imaging Edge virtual camera.
    Uses YUY2 instead of MJPG.
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'YUY2')
        _safe_set(cap, cv2.CAP_PROP_FOURCC, fourcc)
    except Exception as e:
        print(f"[Camera] WARN setting YUY2: {e}", flush=True)

    _safe_set(cap, cv2.CAP_PROP_FRAME_WIDTH,  width)
    _safe_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, height)
    _safe_set(cap, cv2.CAP_PROP_FPS, fps)

    ok, _ = _test_read(cap)
    if not ok:
        print("[Camera] WARN: first read failed after applying caps", flush=True)

def _open_by_name_dshow(name, width, height, fps):
    try:
        for raw in (f"video={name}", name):
            print(f"[Probe] Try DSHOW by name: '{raw}'", flush=True)
            cap = cv2.VideoCapture(raw, cv2.CAP_DSHOW)
            if not (cap and cap.isOpened()):
                if cap is not None: cap.release()
                print("         -> not opened", flush=True)
                continue
            _apply_common_caps(cap, width, height, fps)
            ok, _ = _test_read(cap)
            if ok:
                print("[Probe]   -> OK (DSHOW by name)", flush=True)
                return cap
            cap.release()
            print("         -> read failed after set", flush=True)
    except Exception as e:
        print(f"[Probe] DSHOW name open exception: {e}")
    return None

def _open_by_index(index, api, width, height, fps):
    cap = None
    try:
        print(f"[Probe] Try index={index} api={api}", flush=True)
        cap = cv2.VideoCapture(index, api)
        if not (cap and cap.isOpened()):
            if cap is not None: cap.release()
            print("         -> not opened", flush=True)
            return None
        _apply_common_caps(cap, width, height, fps)
        ok, _ = _test_read(cap)
        if not ok:
            cap.release()
            print("         -> read failed after set", flush=True)
            return None
        print("         -> OK", flush=True)
        return cap
    except Exception as e:
        print(f"         -> exception: {e}", flush=True)
        try:
            if cap is not None: cap.release()
        except:
            pass
        return None

def _probe_and_open(preferred_index, width, height, fps):
    sysn = platform.system()
    api_order = []

    if sysn == "Windows":
        # Prefer MSMF first for Sony Imaging Edge or any virtual cam
        if hasattr(cv2, "CAP_MSMF"):  api_order.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"): api_order.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_ANY"):   api_order.append(cv2.CAP_ANY)
        api_order.append(0)
    elif sysn == "Linux":
        if hasattr(cv2, "CAP_V4L2"):  api_order.append(cv2.CAP_V4L2)
        if hasattr(cv2, "CAP_ANY"):   api_order.append(cv2.CAP_ANY)
        api_order.append(0)
    else:
        if hasattr(cv2, "CAP_ANY"):   api_order.append(cv2.CAP_ANY)
        api_order.append(0)

    print(f"[Probe] Starting camera scan (preferred index {preferred_index})", flush=True)

    # Try by name if explicitly set
    if platform.system() == "Windows" and CAM_NAME:
        cap = _open_by_name_dshow(CAM_NAME, width, height, fps)
        if cap is not None:
            return f"name:{CAM_NAME}", cap

    # Try all backends starting with MSMF
    for api in api_order:
        cap = _open_by_index(preferred_index, api, width, height, fps)
        if cap is not None:
            return preferred_index, cap

    # Fallback: probe other indexes
    for i in range(0, 10):
        if i == preferred_index:
            continue
        for api in api_order:
            cap = _open_by_index(i, api, width, height, fps)
            if cap is not None:
                print(f"[Probe] Using index {i}", flush=True)
                return i, cap

    print("[Probe] No camera could be opened.", flush=True)
    return None, None

class CameraGrabber:
    def __init__(self, index, width, height, fps):
        self.pref_index = index
        self.width = width
        self.height = height
        self.fps = max(1, int(fps))
        self.cap = None
        self.source = None
        self.lock = threading.Lock()
        self.latest = None
        self.running = False
        self.t = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        frame_time = 1.0 / self.fps
        backoff = 0.3
        last_log = time.time()
        while self.running:
            try:
                if self.cap is None:
                    self.source, self.cap = _probe_and_open(self.pref_index, self.width, self.height, self.fps)
                    if self.cap is None:
                        if time.time() - last_log > 2:
                            print("[Camera] Waiting for camera...", flush=True)
                            last_log = time.time()
                        time.sleep(min(5.0, backoff))
                        backoff = min(5.0, backoff * 1.5 + 0.1)
                        continue
                    print(f"[Camera] Active source: {self.source}", flush=True)
                    backoff = 0.3

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("[Camera] Read failed; reopening...", flush=True)
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    time.sleep(0.2)
                    continue

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                with self.lock:
                    self.latest = frame

                time.sleep(frame_time)
            except Exception as e:
                print(f"[Camera] Loop error: {e}\n{traceback.format_exc()}", flush=True)
                try:
                    if self.cap is not None:
                        self.cap.release()
                except:
                    pass
                self.cap = None
                time.sleep(0.5)

    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass


def capture_frame(path="vision_frame.jpg"):
    _, frame = capture_one_frame(timeout_sec=2.0, save_path=None)
    cv2.imwrite(path, frame)
    return path

def analyze_frame(image_path, user_question: str):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    if USE_LLM == "ollama":
        messages = [
            {"role":"system","content":"You are a vision assistant for a robot. Be concise."},
            {"role":"user","content":[
                {"type":"text","text":user_question},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
            ]}
        ]
        resp = oll.chat_vision(messages)
        return oll.extract_text(resp)
    else:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a vision assistant for a robot. Be concise."},
                {"role":"user","content":[
                    {"type":"text","text":user_question},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
                ]}
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

# ================== VOICE ==================
def record_and_transcribe(seconds: int = 15, samplerate: int = 16000) -> str:
    print(f"🎙️ Recording {seconds}s... speak now!")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    print("⏹️ Transcribing...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        with open(tmp.name, "rb") as f:
            resp = oai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
    return (resp.text or "").strip()

def speak(text: str, voice: str = "alloy"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            with oai.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                response_format="wav"
            ) as resp:
                resp.stream_to_file(tmp.name)
            with wave.open(tmp.name, "rb") as wf:
                samplerate = wf.getframerate()
                nch = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            if sampwidth != 2:
                print(f"[TTS] Unexpected sample width: {sampwidth} bytes")
                return
            audio = np.frombuffer(frames, dtype=np.int16)
            if nch == 2:
                audio = audio.reshape(-1, 2)
            sd.play(audio, samplerate=samplerate, blocking=True)
    except Exception as e:
        print(f"[TTS] Could not speak: {e}")

def play_ding():
    try:
        fs = 16000
        duration = 0.25
        f = 1000
        t = np.linspace(0, duration, int(fs*duration), False)
        tone = 0.2 * np.sin(2*np.pi*f*t)
        sd.play(tone.astype(np.float32), fs, blocking=True)
    except Exception as e:
        print(f"[DING] Could not play ding: {e}")

GRABBER = None

def start_camera_stream():
    global GRABBER
    if GRABBER is None:
        print(f"[MCP] Starting camera stream… (pref index={CAM_INDEX}, size={TARGET_W}x{TARGET_H}, fps={CAM_FPS})", flush=True)
        GRABBER = CameraGrabber(index=CAM_INDEX, width=TARGET_W, height=TARGET_H, fps=CAM_FPS)
        GRABBER.start()

def capture_one_frame(timeout_sec=2.0, save_path=None):
    if GRABBER is None:
        raise RuntimeError("Camera not started. Call start_camera_stream() first.")
    t0 = time.time()
    frame = None
    while time.time() - t0 < timeout_sec:
        frame = GRABBER.read()
        if frame is not None:
            break
        time.sleep(0.02)
    if frame is None:
        raise RuntimeError("No frame available from camera (timeout). Is another app using it?")
    if save_path:
        cv2.imwrite(save_path, frame)
        return save_path, frame
    return None, frame

# ================== HTTP HOOK ==================
def _coerce_amount(amount):
    if isinstance(amount, str):
        s = amount.strip().lower()
        return "all" if s == "all" else int(s) if s.isdigit() else 1
    if isinstance(amount, (int, np.integer)):
        return int(amount)
    return 1

# NOTE: legacy/unused — leftover from an earlier MCP-based iteration. Not called
# anywhere in the current pipeline; kept for reference.
def send_color_to_processor(color: str, amount, order: str | None = None) -> bool:
    payload = {
        "action": "pick",
        "color": str(color).lower().strip().replace(" ", "_"),
        "amount": _coerce_amount(amount)
    }
    if order:
        payload["order"] = order
    try:
        r = requests.post(COLOR_PROCESSOR_URL, json=payload, timeout=10)
        try:
            data = r.json()
        except ValueError:
            data = {}
        req_amount = data.get("amount", payload.get("amount"))
        available  = data.get("available")
        if r.status_code == 201 and req_amount and available is not None:
            speak("requested to pick " + str(req_amount) + " but there is "+ str(available) + " coins available, picking whats available")
        if r.status_code == 400:
            speak("There is no coins in this color")
        ok = 200 <= r.status_code < 300
        print(f"[COLOR] POST {COLOR_PROCESSOR_URL} {payload} -> {r.status_code} {r.text[:200]}")
        return ok
    except requests.RequestException as e:
        speak("i failed to pick the " + payload.get("color")+ " color.")
        print(f"[COLOR] Failed to POST color/amount: {e}")
        return False

wake_variants = ["ok robot", "hey robot", "okay robot", "ok, robot", "okay, robot", "okay robert", "ok robert", "okay, robert", "okay rowboat", "ok rowboat", "okay robo"]

def wait_for_wakeword():
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model, 16000)
    q = queue.Queue()
    def callback(indata, frames, time_, status):
        q.put(bytes(indata))
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=callback):
        print("🎙️ Listening for wake word 'ok robot' ...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "").lower()
            if text:
                print(" heard:", text)
                if any(w in text for w in wake_variants):
                    print("🟢 Wake word detected")
                    play_ding()
                    return

def capture_command(max_duration=10, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    samplerate = 16000
    block_ms = 30
    block_size = int(samplerate * block_ms / 1000)
    q = queue.Queue()
    audio_buf = []
    def callback(indata, frames, time_, status):
        q.put(bytes(indata))
    with sd.RawInputStream(samplerate=samplerate, blocksize=block_size,
                           channels=1, dtype="int16", callback=callback):
        print("🎙️ Listening for your command...")
        silence_count = 0
        while True:
            frame = q.get()
            is_speech = vad.is_speech(frame, samplerate)
            audio_buf.append(frame)
            if not is_speech:
                silence_count += 1
            else:
                silence_count = 0
            if silence_count > 100 or len(audio_buf) * block_ms > max_duration * 1000:
                print("⏹️ Command finished, sending to STT...")
                pcm = b"".join(audio_buf)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                    with wave.open(tmpf.name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(samplerate)
                        wf.writeframes(pcm)
                    with open(tmpf.name, "rb") as f:
                        resp = oai.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=f
                        )
                return (resp.text or "").strip()

# ================== INSTRUCTION PARSER ==================
def instruction_to_command(user_instruction: str, frame=None ):
    img_part = []
    if frame is not None:
        path = "frame_for_llm.jpg"
        cv2.imwrite(path, frame)
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_part = [{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}]

    system_prompt = (
        "You are an assistant for a robotics project.\n"
        "Convert the user's instruction into compact JSON with this schema:\n"
        "{\n"
        "  \"action\": \"pick\",\n"
        "  \"items\": [ {\"color\":\"<lowercase>\", \"amount\": <int | \"all\">, \"order\": <optional>, \"box\": <optional color> }, ... ]\n"
        "}\n"
        "Rules:\n"
        "• Always set action to 'pick'.\n"
        "• Always return an 'items' array (even for one color).\n"
        "• Colors: red, green, blue, purple, orange, yellow, black, white, gray.\n"
        "• If the user says 'all', amount='all'. Accept numeric words and digits.\n"
        "• Orders: left_to_right | right_to_left | top_to_bottom | bottom_to_top | largest_first | smallest_first.\n"
        "• If the user says 'in the <color> box', set that item's \"box\" to that color token.\n"
        "• Output ONLY the JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type":"text","text":user_instruction}] + img_part}
    ]
    try:
        if USE_LLM == "ollama":
            resp = oll.chat_text(messages) if not img_part else oll.chat_vision(messages)
            raw = oll.extract_text(resp)
        else:
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print("[LLM] Fallback parser:", e)
        import re
        t = (user_instruction or "").lower()
        color_vocab = ["red","green","blue","yellow","orange","purple","black","white","gray"]
        num_words = {
            "one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
            "eins":1,"zwei":2,"drei":3,"vier":4,"fuenf":5,"fünf":5,"sechs":6,"sieben":7,"acht":8,"neun":9,"zehn":10
        }
        order_map = [
            (r"left\s*to\s*right|links.*(nach|to)\s*rechts",      "left_to_right"),
            (r"right\s*to\s*left|rechts.*(nach|to)\s*links",      "right_to_left"),
            (r"top\s*to\s*bottom|oben.*(nach|to)\s*unten",        "top_to_bottom"),
            (r"bottom\s*to\s*top|unten.*(nach|to)\s*oben",        "bottom_to_top"),
            (r"largest\s*first|biggest\s*first|größ.*zuerst",     "largest_first"),
            (r"smallest\s*first|kleinst.*zuerst",                 "smallest_first"),
        ]
        def detect_order(local_text: str):
            for pat, val in order_map:
                if re.search(pat, local_text):
                    return val
            return None
        def coerce_amount(tok):
            if isinstance(tok, str) and tok in {"all","alle","alles"}:
                return "all"
            if isinstance(tok, str) and tok in num_words:
                return num_words[tok]
            try:
                return int(tok)
            except:
                return 1
        items = []
        used_spans = []
        num_re = r"(?:(\d+)|(" + "|".join(map(re.escape, num_words.keys())) + r")|all|alle|alles)"
        col_re = r"(" + "|".join(map(re.escape, color_vocab)) + r")"
        def add_item_from_match(color_str, amount_val, right_context_start):
            amount_v = coerce_amount(amount_val)
            window = t[right_context_start:right_context_start+120]
            ord_val = detect_order(window)
            items.append({"color": color_str, "amount": amount_v, **({"order": ord_val} if ord_val else {})})
        pat1 = re.compile(rf"\b{num_re}\s+{col_re}\b")
        for m in pat1.finditer(t):
            amt_digit, amt_word, col = m.groups()
            tok = m.group(0).split()[0]
            amt = ("all" if tok in {"all","alle","alles"} else (amt_digit or amt_word))
            add_item_from_match(col, amt, m.end())
            used_spans.append(m.span())
        pat2 = re.compile(rf"\b{col_re}\s+{num_re}\b")
        for m in pat2.finditer(t):
            col, amt_digit, amt_word = m.groups()
            tok = m.group(0).split()[-1]
            amt = ("all" if tok in {"all","alle","alles"} else (amt_digit or amt_word))
            add_item_from_match(col, amt, m.end())
            used_spans.append(m.span())
        global_all = re.search(r"\b(all|alle|alles)\b", t) is not None
        for m in re.finditer(rf"\b{col_re}\b", t):
            col = m.group(0)
            if any(m.start() >= s and m.end() <= e for (s, e) in used_spans):
                continue
            window = t[m.end():m.end()+120]
            ord_val = detect_order(window)
            amt = "all" if global_all else 1
            items.append({"color": col, "amount": amt, **({"order": ord_val} if ord_val else {})})
        mbox = re.search(r"in\s+the\s+(red|green|blue)\s+box", t)
        if items and mbox:
            for it in items:
                it["box"] = mbox.group(1)
        if not items:
            return None
        return {"action":"pick","items":items}

def generate_response_from_command(command: dict, user_instruction: str) -> str:
    try:
        if USE_LLM == "ollama":
            messages = [
                {"role":"system","content":"You are the voice of a helpful robot assistant. Be concise and natural."},
                {"role":"user","content":f"User said: '{user_instruction}'.\nParsed command: {json.dumps(command)}.\nReply with what the robot should say out loud."}
            ]
            resp = oll.chat_text(messages)
            return oll.extract_text(resp) or "Okay, executing your command."
        else:
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are the voice of a helpful robot assistant. Be concise and natural."},
                    {"role":"user","content":f"User said: '{user_instruction}'.\nParsed command: {json.dumps(command)}.\nReply with what the robot should say out loud."}
                ],
                max_tokens=60, temperature=0.7
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Could not generate spoken response: {e}")
        return "Okay, executing your command."

# ================== ROBOT HOOK ==================
def write_code_and_run_robot(coords: dict | None = None):
    if coords is not None:
        with open(COORDS_FILE, "w") as f:
            json.dump(coords, f)
        print(f"[MCP] Wrote coords to {COORDS_FILE}: {coords}")

# ================== PROCESSOR ==================


def process_items_with_opencv(items, frame):
    """
    Select coins per item and attach matching box centers in mm if requested.
    Coin and box coordinates are converted into plate coordinates (mm),
    with origin at the TOP orange reference rectangle.

    IMPORTANT:
    - We first detect coins and boxes in pixels.
    - We compute the plate transform from the two orange reference markers.
    - For EACH detection we convert (u,v) -> (X_mm, Y_mm).
    - We keep ONLY those with 0 <= X_mm <= 340 and 0 <= Y_mm <= 340.
    """
    combined_batch = []

    # --- detect once (pixel domain) ---
    coin_detections = detect_on_frame(frame.copy())     # coins on full frame
    boxes_pixel     = detect_boxes_by_color(frame.copy())  # boxes in full frame
    ref             = detect_reference_markers(frame.copy())  # orange corner markers

    if not ref:
        print("[REF] Could not detect reference markers.")
        speak("I cannot see the orange reference markers on the plate.")
        return []

    tf = compute_plate_transform(ref)
    if tf is None:
        print("[REF] Invalid reference geometry.")
        speak("The reference markers are not clear enough to compute coordinates.")
        return []

    # --- keep only coins INSIDE 340x340 mm plate, and assign x_mm,y_mm ---
    filtered_coins = []
    for d in coin_detections:
        cx = d["cx"]
        cy = d["cy"]
        inside, X_mm, Y_mm = inside_plate_mm(cx, cy, tf)
        if not inside:
            continue
        d["x_mm"] = X_mm
        d["y_mm"] = Y_mm
        filtered_coins.append(d)
    coin_detections = filtered_coins

    # --- keep only boxes INSIDE plate, and assign x_boxmm,y_boxmm ---
    boxes = {}
    for color, info in boxes_pixel.items():
        xc = info["xc"]
        yc = info["yc"]
        inside, X_mm, Y_mm = inside_plate_mm(xc, yc, tf)
        if not inside:
            continue
        info["x_boxmm"] = X_mm
        info["y_boxmm"] = Y_mm
        boxes[color] = info

    # --- selection per item (same as before, but only using filtered coins) ---
    used_ids = set()
    for it in items:
        color     = (it.get("color") or "").lower()
        amount    = it.get("amount", "all")
        order     = it.get("order", "left_to_right")
        box_color = (it.get("box") or "").lower()

        same_color = [
            d for d in coin_detections
            if d["color"] == color and d["id"] not in used_ids
        ]

        if order == "right_to_left":
            same_color.sort(key=lambda d: d["bbox"][0], reverse=True)
        elif order == "left_to_right":
            same_color.sort(key=lambda d: d["bbox"][0])
        elif order == "bottom_to_top":
            same_color.sort(key=lambda d: d["bbox"][1], reverse=True)
        elif order == "top_to_bottom":
            same_color.sort(key=lambda d: d["bbox"][1])
        elif order == "largest_first":
            same_color.sort(key=lambda d: d["area_px"], reverse=True)
        elif order == "smallest_first":
            same_color.sort(key=lambda d: d["area_px"])

        k = len(same_color) if str(amount) == "all" else int(amount)
        sel = same_color[:max(0, min(k, len(same_color)))]

        x_boxmm = y_boxmm = None
        if box_color in boxes:
            x_boxmm = boxes[box_color]["x_boxmm"]
            y_boxmm = boxes[box_color]["y_boxmm"]
        elif box_color:
            print(f"[WARN] Box color '{box_color}' not found inside plate.")

        for d in sel:
            used_ids.add(d["id"])
            entry = {
                "x_mm":  d["x_mm"],
                "y_mm":  d["y_mm"],
                "color": d["color"],
                "box":   box_color
            }
            if x_boxmm is not None and y_boxmm is not None:
                entry["x_boxmm"] = x_boxmm
                entry["y_boxmm"] = y_boxmm
            combined_batch.append(entry)

    # --- safety check unchanged ---
    out_of_reach = []
    for it in combined_batch:
        x_box = it.get("x_boxmm", 0)
        y_box = it.get("y_boxmm", 0)
        max_y_allowed = 360.0 if x_box > 225 else 400.0
        if y_box > max_y_allowed:
            out_of_reach.append(it)

    if out_of_reach:
        print("[SAFETY] Box out of reach (dynamic check):", out_of_reach)
        speak("The target box is out of reach based on its position. I will not execute this action.")
        return []

    if combined_batch:
        send_coords_batch(combined_batch)
    return combined_batch



# ================== ORCHESTRATOR ==================
class MCP:
    def handle_command(self, command):
        print(f"[MCP] Command: {command}")
        action = (command or {}).get("action")

        items = (command or {}).get("items")
        if not items:
            picks = (command or {}).get("picks")
            if isinstance(picks, list) and picks:
                items = picks
            else:
                color  = (command or {}).get("color")
                amount = (command or {}).get("amount")
                if color:
                    items = [{"color": color, "amount": amount if amount is not None else 1}]

        default_order = (command or {}).get("order")

        if action != "pick":
            print("[MCP] Unknown action. Only 'pick' supported.")
            return
        if not items:
            print("[MCP] No items provided.")
            return

        payload_items = []
        for i, it in enumerate(items, start=1):
            c = (it or {}).get("color")
            if not c:
                print(f"[MCP] Skipping item #{i}: missing color.")
                continue
            payload_items.append({
                "color": c,
                "amount": it.get("amount", 1),
                **({"order": (it.get("order") or default_order)} if (it.get("order") or default_order) else {}),
                **({"box": it.get("box")} if it.get("box") else {})
            })

        if payload_items:
            _, frame = capture_one_frame()
            coords = process_items_with_opencv(payload_items, frame)
            if coords:
                print("[MCP] Sent coords:", coords)

def preview_loop():
    try: cv2.namedWindow("Live Preview", cv2.WINDOW_NORMAL)
    except Exception as e:
        print(f"[Preview] GUI init failed: {e}", flush=True); return
    while True:
        frame = GRABBER.read() if GRABBER else None
        if frame is None: time.sleep(0.2); continue
        annotated = frame.copy()
        annotated = _annotate_colors_on_frame(annotated)
        annotated = _annotate_boxes_on_frame(annotated)
        annotated = _annotate_reference_markers_on_frame(annotated)  # <- new
        try:
            cv2.imshow("Live Preview", annotated)
            if cv2.waitKey(1) & 0xFF == 27: break
        except cv2.error as e:
            print(f"[Preview] GUI error: {e}", flush=True); time.sleep(0.5)


def main():
    threading.Thread(target=preview_loop,daemon=True).start()
    start_camera_stream()
    mcp = MCP()
    print("Say 'ok robot' to wake me up. After the ding, say your command.")
    while True:
        wait_for_wakeword()
        user_instruction = capture_command()
        if not user_instruction: continue
        if user_instruction.lower() in {"exit","quit","stop"}: break
        print(f"User: {user_instruction}")
        if any(q in user_instruction.lower() for q in ["see","look","camera","show"]):
            try:
                frame_path = capture_frame()
                analysis = analyze_frame(frame_path, user_instruction)
                print("🔎 Vision:", analysis); speak(analysis)
            except Exception as e:
                print("❌ Vision analysis failed:", e); speak("Sorry, I could not analyze the camera frame.")
            continue
        command = instruction_to_command(user_instruction)
        if command:
            response_text = generate_response_from_command(command, user_instruction)
            speak(response_text)
            mcp.handle_command(command)
        else:
            speak("Please tell me to pick a coin. Say the color and amount you want.")

if __name__=="__main__":
    main()
