import os
import socket
import time
from flask import Flask, request, jsonify
import threading
from collections import deque

UR_IP = os.getenv("UR_ROBOT_IP", "<robot-ip>")
DASHBOARD_PORT = 29999
SCRIPT_PORT = 30002
INTERPRETER_PORT = 30020

HOST = os.getenv("LOCAL_SERVER_HOST", "<server-ip>")
PORT = 8000

app = Flask(__name__)
ROBOT_TIMEOUT = 60 

# ================== Queue/Stack for jobs ==================
# False = FIFO (queue), True = LIFO (stack)
STACK_MODE = False

# Queue now holds **job IDs** (ints). Actual jobs live in job_repository.
job_queue = deque()
queue_lock = threading.Lock()
coords_available = threading.Event()  # "there is at least one job to run"

# Job repository & current job
job_repository = {}                 # job_id -> job dict
job_id_lock = threading.Lock()
current_job_id = None               # id of job currently running (or None)

# Robot sync
robot_has_signaled = threading.Event()

# ================== Job helpers ==================
def new_job_id() -> int:
    with job_id_lock:
        return len(job_repository) + 1

def enqueue_job(job: dict) -> int:
    """Store job in repository and push its id into the queue."""
    with job_id_lock:
        job_id = job["id"]
        job_repository[job_id] = job
    with queue_lock:
        if STACK_MODE:
            job_queue.append(job_id)
        else:
            job_queue.append(job_id)
        coords_available.set()
        qsz = len(job_queue)
    return qsz

# ================== HTTP endpoints ==================
@app.route('/robot_signal', methods=['POST'])
def robot_signal():
    # Generic sync signal (used by pick-cycle phases and plain URP jobs)
    robot_has_signaled.set()

    # If a URP program job is running, this likely means "program finished".
    with job_id_lock:
        jid = current_job_id
    if jid:
        job = job_repository.get(jid)
        if job and job.get("type") == "urp_program" and job.get("status") == "running":
            job["status"] = "finished"
    return '', 200

@app.route('/coords', methods=['POST'])
def set_coords():
    """
    Accepts:
      - Batch only:
        {
          "coords":[
             {
               "x_mm": 12.3,
               "y_mm": -45.0,
               "color": "red",
               "box": "top" | "middle" | "bottom" | "blue" | ... ,  # optional label
               "x_boxmm": 111.1, "y_boxmm": 222.2                   # optional dynamic place coords
             },
             ...
          ]
        }

    Each item is stored as a 4-tuple:
      (x_mm, y_mm, color, box)
      where box can be either a string or a dict {"x_boxmm": float, "y_boxmm": float}.
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    items = []

    if "coords" not in data:
        return jsonify({"error": "No coords found"}), 400

    lst = data["coords"]
    if not isinstance(lst, list) or not lst:
        return jsonify({"error": "'coords' must be a non-empty list"}), 400

    for item in lst:
        try:
            x_val = float(item["x_mm"])
            y_val = float(item["y_mm"])
        except Exception:
            return jsonify({"error": "Each coords item must include numeric x_mm and y_mm"}), 400

        color = str(item.get("color", "")).lower().strip() or None

        # --- check for dynamic box coordinates ---
        bx = item.get("x_boxmm") or item.get("box_x_mm")
        by = item.get("y_boxmm") or item.get("box_y_mm")

        if bx is not None and by is not None:
            try:
                box = {
                    "x_boxmm": float(bx),
                    "y_boxmm": float(by)
                }
            except Exception:
                return jsonify({"error": "x_boxmm/y_boxmm must be numeric"}), 400
        else:
            # fallback: use simple label if coordinates not supplied
            box = str(item.get("box", "")).lower().strip() or None

        items.append((x_val, y_val, color, box, float(bx) if bx is not None else None, float(by) if by is not None else None))

    # --- enqueue the job ---
    jid = new_job_id()
    job = {
        "id": jid,
        "type": "batch_pick",
        "status": "queued",
        "items": items,  # (x_mm, y_mm, color, box)
    }
    qsz = enqueue_job(job)

    print(f"📥 Queued JOB#{jid} (batch_pick) with {len(items)} item(s). Queue size={qsz}.")
    return jsonify({"job_id": jid, "queued_items": len(items), "queue_size": qsz}), 200



# ===== Legacy endpoints kept: run an URP like final_move.urp as a job =====
@app.route("/jobs/coin-extraction", methods=["POST"])
def coin_extraction():
    """
    Enqueue a URP program job (e.g., final_move.urp).
    Response: 201 with {"location": <job_id>}
    """

    jid = new_job_id()
    job = {
        "id": jid,
        "type": "urp_program",
        "program": "final_move.urp",
        "status": "queued",
    }
    enqueue_job(job)

    print(f"📥 Queued JOB#{jid} (urp_program: final_move.urp).")
    return jsonify({"location": jid}), 201

@app.route("/jobs/coin-extraction/<int:job_id>/status", methods=["GET"])
def job_status(job_id: int):
    job = job_repository.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    out = {"status": job.get("status")}
    if job.get("type") == "urp_program":
        out["program"] = job.get("program")
        out["job_id"] = job_id
    elif job.get("type") == "batch_pick":
        out["job_id"] = job_id
    return jsonify(out), 200

# ================== Robot helpers ==================
def get_robot_mode():
    try:
        resp = send_dashboard_command("robotmode")
        # Expect something like "Robotmode: RUNNING"
        if resp.startswith("Robotmode:"):
            return resp.split(":", 1)[1].strip()
    except Exception as e:
        print(f"[WARN] Could not get robot mode: {e}")
    return "UNKNOWN"


def wait_until_robot_post(timeout=ROBOT_TIMEOUT, poll_interval=0.5):
    """
    Wait until robot signals back, OR abort if robot is not running.
    """
    robot_has_signaled.clear()
    start = time.time()

    while True:
        # Case 1: Robot has signaled
        if robot_has_signaled.is_set():
            return

        # Case 2: Timeout
        if (time.time() - start) > timeout:
            raise TimeoutError(f"Robot did not respond within {timeout} seconds")

        # Case 3: Robot stopped
        mode = get_robot_mode()
        print(mode)
        if mode not in ("RUNNING", "PLAYING"):
            raise RuntimeError(f"Aborting: robot mode is {mode}")

        time.sleep(poll_interval)


def send_interpreter_command(script):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UR_IP, INTERPRETER_PORT))
        s.sendall(script.encode())

def send_dashboard_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as dash:
        dash.connect((UR_IP, DASHBOARD_PORT))
        dash.recv(1024)  # discard banner "Connected: Universal Robots Dashboard Server"
        dash.sendall((command + '\n').encode())
        time.sleep(0.1)
        response = dash.recv(1024).decode().strip()
        return response

def send_urscript(script):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UR_IP, SCRIPT_PORT))
        s.sendall(script.encode())
        print(f"[URScript] {len(script)} bytes sent.")

def run_program(name):
    send_dashboard_command(f'load {name}')
    time.sleep(0.5)
    send_dashboard_command('play')
    wait_until_robot_post()

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# ================== Flask server thread ==================
def start_flask_server():
    app.run(host=HOST, port=PORT)

# ================== One pick cycle for a single (x_mm, y_mm) ==================
def execute_pick_cycle(x_mm, y_mm, color, box, x_boxmm, y_boxmm):
    # Common conversion used for camera-plane mm → robot local meters
    x = (float(x_mm) + 12.5 )/ 1000.0
    y = (float(y_mm) + 20 ) / 1000.0
    x_place = (float(x_boxmm)  or 0) / 1000.0
    y_place = (float(y_boxmm)  or 0) / 1000.0

    # Phase 1: go to pick
    urscript = f"""
def move_test():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose1 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.00978, 0.762, 0.017, 0.036])

    movej(p[-0.22025, -0.36187, 0.16515, 1.277, 2.968, -1.184], a=0.8, v=0.8)
    movel(pose1, a=0.8, v=0.5)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test()
"""

    # Phase 2: place
    # If dynamic coords provided, compute dynamic placement; else use legacy fixed boxes
    
        
        

    put_in_dynamic_box = f"""
def move_test2():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.03770, 0.762, 0.017, 0.036])

    
    movel(pose2, a=0.8, v=0.5)
    # dynamic placement pose at same orientation as legacy place moves
    place_pose = pose_trans(final, p[{ -x_place:.6f}, { -y_place:.6f}, -0.05770, 0.762, 0.017, 0.036])
    movel(place_pose, a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
    place_script = put_in_dynamic_box
    
        # legacy fixed targets by label
    put_in_top_box = f"""
def move_test2():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.0377, 0.762, 0.017, 0.036])

    movel(pose2, a=0.8, v=0.5)
    movel(p[-0.50242, -0.45141, -0.01116, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
    put_in_middle_box = f"""
def move_test2():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.0377, 0.762, 0.017, 0.036])

    movel(pose2, a=0.8, v=0.5)
    movel(p[-0.43007, -0.52461, -0.01142, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
    put_in_bottom_box = f"""
def move_test2():

    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.0377, 0.762, 0.017, 0.036])

    movel(pose2, a=0.8, v=0.5)
    movel(p[-0.36961, -0.58394, -0.01161, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
        # optional color-specific legacy (kept)
    put_in_red_box = f"""
def move_test2():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.0377, 0.762, 0.017, 0.036])

    movel(pose2, a=0.8, v=0.5)
    movel(p[-0.53875, -0.18577, -0.01412, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
    put_in_blue_box = f"""
def move_test2():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.0377, 0.762, 0.017, 0.036])

    movel(pose2, a=0.8, v=0.5)
    movel(p[-0.60875, -0.25577, -0.01412, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
    put_in_green_box = f"""
def move_test2():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]
    pose2 = pose_trans(final, p[{ -x:.6f}, { -y:.6f}, -0.0377, 0.762, 0.017, 0.036])

    movel(pose2, a=0.8, v=0.5)
    movel(p[-0.46875, -0.11577, -0.01412, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test2()
"""
        # choose legacy by label
        #if box_label == "top":
            #place_script = put_in_top_box
        #elif box_label == "bottom":
            #place_script = put_in_bottom_box
        #elif box_label == "red":
            #place_script = put_in_red_box
        #elif box_label == "blue":
            #place_script = put_in_blue_box
        #elif box_label == "green":
            #place_script = put_in_green_box
        #else:
            #place_script = put_in_middle_box

    urscript3 = f"""
def move_test3():
    final = p[-0.36405, -0.17993, -0.06523, 1.189, 2.910, 0.003]

    movej(p[-0.22025, -0.36187, 0.16515, 1.277, 2.968, -1.184], a=0.8, v=0.8)

    socket_open("{HOST}", {PORT}, 3.0)
    socket_send_string("POST /robot_signal HTTP/1.1", 3.0)
    socket_close()
end
move_test3()
"""

    # Execute the 3-phase cycle (same as your original logic)
    send_interpreter_command("interpreter_mode()")
    send_urscript(urscript)
    wait_until_robot_post()
    send_interpreter_command("end_interpreter()")

    run_program("grip.urp")

    send_interpreter_command("interpreter_mode()")
    send_urscript(place_script)
    wait_until_robot_post()
    send_interpreter_command("end_interpreter()")

    run_program("ungrip.urp")

    send_interpreter_command("interpreter_mode()")
    send_urscript(urscript3)
    wait_until_robot_post()
    send_interpreter_command("end_interpreter()")

# ================== Main worker ==================
def worker_loop():
    global current_job_id
    while True:
        if not coords_available.is_set():
            print(" Waiting for jobs ...")
        coords_available.wait()

        while True:
            with queue_lock:
                if not job_queue:
                    coords_available.clear()
                    break
                # Pop a job id per STACK_MODE or FIFO
                jid = (job_queue.pop() if STACK_MODE else job_queue.popleft())
                remaining_jobs = len(job_queue)

            job = job_repository.get(jid)
            if not job:
                continue

            # Mark running
            job["status"] = "running"
            with job_id_lock:
                current_job_id = jid

            print(f"▶ Executing JOB#{jid} ({job['type']}). Remaining jobs={remaining_jobs}")

            try:
                if job["type"] == "batch_pick":
                    run_program("move_to_1400.urp")
                    run_program("ungrip.urp")
                    items = job.get("items") or []
                    print(f"   • {len(items)} item(s) in batch")
                    for idx, tpl in enumerate(items, start=1):

                        x_local, y_local, color_local, box_local, bx_local, by_local = (
                            tpl + (None,) * (6 - len(tpl))
                        )    
                        print(f"  - Item {idx}: x={x_local:.3f}, y={y_local:.3f}, color={color_local}, box={box_local}, box_x={bx_local}, box_y={by_local}")
                        try:
                            execute_pick_cycle(
                                x_local, y_local,
                                color=color_local,
                                box=box_local,
                                x_boxmm=bx_local,
                                y_boxmm=by_local
                            )
                        except Exception as e:
                            print(f"❌ Item failed: {e}. Continuing with next in this job.")
                            job["status"] = "failed"


                        
                            
                    if job["status"] == "running":
                        job["status"] = "finished"

                elif job["type"] == "urp_program":
                    # Load & play program; expect /robot_signal on completion
                    send_dashboard_command(f'load final_move.urp')
                    time.sleep(0.5)
                    send_dashboard_command('play')
                    wait_until_robot_post()
                    # If not already flipped by /robot_signal, mark finished now
                    if job.get("status") != "finished":
                        job["status"] = "finished"

                else:
                    print("   ❌ Unknown job type")
                    job["status"] = "failed"

                print(f"✅ JOB#{jid} complete.")

            except Exception as e:
                print(f"❌ JOB#{jid} failed: {e}")
                job["status"] = "failed"

            # Clear current job id
            with job_id_lock:
                current_job_id = None

        print("🌙 Queue empty.")

# ================== Main orchestrator ==================
def main():
    # Start HTTP server
    threading.Thread(target=start_flask_server, daemon=True).start()
    time.sleep(2)
    # Start worker
    worker_loop()

if __name__ == "__main__":
    main()
