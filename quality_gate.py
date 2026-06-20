from flask import Flask, request, jsonify
import cv2, numpy as np, time, threading
from picamera2 import Picamera2

app = Flask(__name__)
expected_color = None

# -------- camera & detection settings --------
TARGET_W, TARGET_H = 640, 480
MIN_RADIUS, MAX_RADIUS, MIN_DIST = 10, 45, 35
HOUGH_DP, HOUGH_P1, HOUGH_P2 = 1.2, 80, 35
ROI_X_RATIO, ROI_Y_RATIO, ROI_W_RATIO, ROI_H_RATIO = 0.3, 0.3, 0.4, 0.4

# shared data between threads
latest_detection = {"color": None, "timestamp": 0}
stop_flag = False


# ---------- color & detection helpers ----------
def classify_color_bgr(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = map(int, hsv)
    if v < 60 or s < 40:
        return "unknown"
    if h < 10 or h >= 170: return "red"
    if 10 <= h < 25: return "orange"
    if 25 <= h < 35: return "yellow"
    if 35 <= h < 85: return "green"
    if 90 <= h < 130: return "blue"
    if 130 <= h < 170: return "purple"
    return "unknown"


def mean_bgr_in_circle(img, cx, cy, r):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    B, G, R = cv2.mean(img, mask=mask)[:3]
    return (int(B), int(G), int(R))


def detect_circles(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
                         minDist=MIN_DIST, param1=HOUGH_P1, param2=HOUGH_P2,
                         minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    if c is None:
        return []
    return np.uint16(np.around(c[0, :])).tolist()


# ---------- continuous camera thread ----------
def camera_loop():
    global latest_detection, stop_flag
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (TARGET_W, TARGET_H)}
    ))
    picam2.start()
    time.sleep(0.5)
    print("[INFO] Continuous camera stream started.")

    x = int(ROI_X_RATIO * TARGET_W)
    y = int(ROI_Y_RATIO * TARGET_H)
    w = int(ROI_W_RATIO * TARGET_W)
    h = int(ROI_H_RATIO * TARGET_H)

    cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)

    while not stop_flag:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ROI box
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # detect in ROI
        roi = bgr[y:y + h, x:x + w]
        circles = detect_circles(roi)

        if circles:
            cx, cy, r = circles[0]
            if MIN_RADIUS <= r <= MAX_RADIUS:
                cx_full, cy_full = cx + x, cy + y
                color_bgr = mean_bgr_in_circle(bgr, int(cx_full), int(cy_full), int(r))
                color_name = classify_color_bgr(color_bgr)
                latest_detection = {"color": color_name, "timestamp": time.time()}

                # ----- draw detection overlay -----
                if color_name != "unknown":
                    cv2.circle(bgr, (int(cx_full), int(cy_full)), int(r), (0, 255, 0), 2)
                    cv2.putText(
                        bgr, color_name, (int(cx_full - r), int(cy_full - r - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

        cv2.imshow("Live Camera", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_flag = True
            break

    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    print("[INFO] Camera thread stopped.")


# start background thread immediately
threading.Thread(target=camera_loop, daemon=True).start()


# ---------- API endpoints ----------
@app.route("/set_expected_color", methods=["POST"])
def set_expected_color():
    global expected_color
    data = request.get_json(force=True)
    expected_color = data.get("color")
    if not expected_color:
        return jsonify({"error": "no color provided"}), 400
    print(f"[INFO] Expected color set to: {expected_color}")
    return jsonify({"status": "ok", "expected_color": expected_color}), 200


@app.route("/check_color", methods=["GET"])
def check_color():
    """Return the most recent valid detection."""
    global expected_color
    if expected_color is None:
        return jsonify({"error": "expected color not set"}), 400

    start = time.time()
    print("[INFO] Waiting for coin detection event...")
    while time.time() - start < 10:
        det = latest_detection.copy()
        if det["timestamp"] > start:
            color = det["color"]
            print(f"[INFO] Latest detection: {color}")
            if color == "unknown":
                return jsonify({"error": "no valid coin detected"}), 404
            match = color == expected_color
            result = {"detected": color, "expected": expected_color}
            return (jsonify(result), 200 if match else 404)
        time.sleep(0.1)

    return jsonify({"error": "timeout - no coin detected"}), 404


@app.route("/shutdown", methods=["GET"])
def shutdown():
    global stop_flag
    stop_flag = True
    return "Camera shutting down", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
