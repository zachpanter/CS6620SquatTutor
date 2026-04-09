import cv2
import mediapipe as mp
import numpy as np
import time
import queue
import typer
import json
import os
from datetime import datetime
from rich.console import Console

app = typer.Typer()
console = Console()

# --- CONSTANTS ---
left_ear, right_ear = 7, 8
left_shoulder, left_waist, left_knee, left_ankle = 11, 23, 25, 27
right_shoulder, right_waist, right_knee, right_ankle = 12, 24, 26, 28


# --- MATH & GEOMETRY FUNCTIONS ---
def calc_angle_2d(a, b, c):
    angle_ba = np.array([a.x - b.x, a.y - b.y])
    angle_bc = np.array([c.x - b.x, c.y - b.y])
    length_ba = np.linalg.norm(angle_ba)
    length_bc = np.linalg.norm(angle_bc)
    if length_ba == 0 or length_bc == 0: return 0.0
    cosine_angle = np.dot(angle_ba, angle_bc) / (length_ba * length_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calc_angle_3d(a, b, c):
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    length_ba, length_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if length_ba == 0 or length_bc == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / (length_ba * length_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calc_vertical_angle(top, bottom):
    vec = np.array([top.x - bottom.x, top.y - bottom.y])
    vert = np.array([0, -1])
    length_vec = np.linalg.norm(vec)
    if length_vec == 0: return 0.0
    cosine_angle = np.dot(vec, vert) / (length_vec * np.linalg.norm(vert))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def check_visibility(landmarks, indices, threshold=0.6):
    for idx in indices:
        if landmarks[idx].visibility < threshold: return False
    return True


def draw_skeleton(image, landmarks):
    h, w, _ = image.shape
    connections = [
        (left_shoulder, right_shoulder), (left_waist, right_waist),
        (right_shoulder, right_waist), (left_shoulder, left_waist),
        (right_waist, right_knee), (left_waist, left_knee),
        (right_knee, right_ankle), (left_knee, left_ankle)
    ]
    joints_to_draw = [
        left_shoulder, right_shoulder, left_waist, right_waist,
        left_knee, right_knee, left_ankle, right_ankle
    ]
    color, thickness, radius = (245, 117, 66), 5, 5

    for start_index, end_index in connections:
        start, end = landmarks[start_index], landmarks[end_index]
        cv2.line(image, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), color, thickness)

    for joint in joints_to_draw:
        lm = landmarks[joint]
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), radius, color, thickness)


# --- TRACKING SESSION WRAPPER ---
def run_tracking_session(focus_metric: str) -> list:
    BaseOptions = mp.tasks.BaseOptions
    MODEL_PATH = './pose_landmarker_full.task'
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode

    rendered_frames_queue = queue.Queue(maxsize=3)
    pending_frames = {}

    metric_data = []
    session_start = time.time()
    tracking_state = {"started": False}

    def handle_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if timestamp_ms not in pending_frames: return
        frame_to_draw = pending_frames.pop(timestamp_ms)
        h, w, _ = frame_to_draw.shape

        current_time = time.time()
        elapsed = current_time - session_start

        # 1. ALWAYS draw the skeleton if visible, even during the countdown
        if result.pose_landmarks:
            draw_skeleton(frame_to_draw, result.pose_landmarks[0])

        # 2. Handle the Timing Phases independently of the landmarks
        if elapsed < 8.0:
            # --- PHASE 1: COUNTDOWN ---
            countdown_val = int(8.0 - elapsed) + 1
            text_size = cv2.getTextSize(f"{countdown_val}", cv2.FONT_HERSHEY_SIMPLEX, 5.0, 10)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame_to_draw, f"{countdown_val}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 255), 10, cv2.LINE_AA)

            label_size = cv2.getTextSize("GET INTO POSITION", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            label_x = (w - label_size[0]) // 2
            cv2.putText(frame_to_draw, "GET INTO POSITION", (label_x, text_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            if focus_metric in ["valgus", "stance"]:
                orientation_msg = "Recommendation: FACE THE CAMERA DIRECTLY"
            else:
                orientation_msg = "Recommendation: FACE PERPENDICULAR (SIDE) TO CAMERA"

            orient_size = cv2.getTextSize(orientation_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            orient_x = (w - orient_size[0]) // 2
            cv2.putText(frame_to_draw, orientation_msg, (orient_x, text_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        else:
            # --- PHASE 2: ACTIVE TRACKING ---
            if not tracking_state["started"]:
                print('\a', end='', flush=True)
                tracking_state["started"] = True

            # Only calculate and display text metrics if landmarks are currently visible
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                display_text = ""

                if focus_metric == "knee_depth" and check_visibility(landmarks, [left_waist, left_knee, left_ankle]):
                    val = calc_angle_3d(landmarks[left_waist], landmarks[left_knee], landmarks[left_ankle])
                    display_text = f"Knee Depth: {int(val)} deg"
                    metric_data.append((current_time, val))

                elif focus_metric == "valgus" and check_visibility(landmarks,
                                                                   [left_waist, left_knee, left_ankle, right_waist,
                                                                    right_knee, right_ankle]):
                    l_valgus = calc_angle_2d(landmarks[left_waist], landmarks[left_knee], landmarks[left_ankle])
                    r_valgus = calc_angle_2d(landmarks[right_waist], landmarks[right_knee], landmarks[right_ankle])
                    val = (l_valgus + r_valgus) / 2
                    display_text = f"Avg Valgus: {int(val)} deg"
                    metric_data.append((current_time, val))

                elif focus_metric == "back_lean" and check_visibility(landmarks, [left_shoulder, left_waist]):
                    val = calc_vertical_angle(landmarks[left_shoulder], landmarks[left_waist])
                    display_text = f"Back Lean: {int(val)} deg"
                    metric_data.append((current_time, val))

                elif focus_metric == "stance" and check_visibility(landmarks, [left_ankle, right_ankle]):
                    l_ank = np.array([landmarks[left_ankle].x * w, landmarks[left_ankle].y * h])
                    r_ank = np.array([landmarks[right_ankle].x * w, landmarks[right_ankle].y * h])
                    val = np.linalg.norm(l_ank - r_ank)
                    display_text = f"Stance Width: {int(val)} px"
                    metric_data.append((current_time, val))

                if display_text:
                    cv2.putText(frame_to_draw, display_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                cv2.LINE_AA)

        if not rendered_frames_queue.full():
            rendered_frames_queue.put(frame_to_draw)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            timestamp_ms = int(time.time() * 1000)
            pending_frames[timestamp_ms] = frame.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, timestamp_ms)

            for k in [ts for ts in pending_frames.keys() if ts < timestamp_ms - 1000]:
                del pending_frames[k]

            try:
                display_frame = rendered_frames_queue.get_nowait()
                cv2.imshow(f"Biomechanics Dashboard - Focus: {focus_metric.upper()}", display_frame)
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.1)

        # --- PHASE 3: PURGE & RETURN ---
    cutoff_time = time.time() - 8.0
    clean_data = [val for ts, val in metric_data if ts <= cutoff_time]
    return clean_data


# --- EVALUATION LOGIC ---
def evaluate_metric(metric: str, data: list, experience: str) -> dict:
    if not data:
        return {"passed": False, "actual": "No Data", "target": "N/A", "raw_data": []}

    clean_data = [float(x) for x in data]
    result = {"raw_data": clean_data}

    if metric == "back_lean":
        actual = max(clean_data)
        target = 35 if experience == "experienced" else 45
        result.update({"passed": bool(actual <= target), "actual": f"{int(actual)} deg", "target": f"<= {target} deg"})

    elif metric == "valgus":
        actual = min(clean_data)
        target = 130
        result.update({"passed": bool(actual >= target), "actual": f"{int(actual)} deg", "target": f">= {target} deg"})

    elif metric == "knee_depth":
        actual = min(clean_data)
        target = 80 if experience == "experienced" else 90
        result.update({"passed": bool(actual <= target), "actual": f"{int(actual)} deg", "target": f"<= {target} deg"})

    elif metric == "stance":
        actual = float(np.std(clean_data))
        target = 10 if experience == "experienced" else 20
        result.update({"passed": bool(actual <= target), "actual": f"{int(actual)} px variance",
                       "target": f"<= {target} px variance"})

    return result


def save_results(log_data: dict):
    filename = "biomechanics_log.json"
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            pass

    existing_data.append(log_data)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)
    console.print(f"[dim]Results saved to {filename}[/dim]")


# --- TYPER CLI ---
@app.command()
def main():
    console.print("Starting Sequential Biomechanics Tracker")

    experience = typer.prompt("Are you a [beginner] or [experienced] lifter?", default="beginner").lower()
    sequence = ["back_lean", "valgus", "knee_depth", "stance"]

    session_log = {
        "timestamp": datetime.now().isoformat(),
        "experience": experience,
        "results": {}
    }

    for metric in sequence:
        display_name = metric.replace('_', ' ').title()

        proceed = typer.confirm(
            f"\nReady for: {display_name}? (Press Enter to start, 'q' in the window to quit)",
            default=True
        )
        if not proceed:
            console.print("Session aborted by user.")
            break

        console.print(f"Tracking {display_name}...")
        data = run_tracking_session(focus_metric=metric)

        eval_result = evaluate_metric(metric, data, experience)
        session_log["results"][metric] = eval_result

        console.print(f"Target: {eval_result['target']} | Actual: {eval_result['actual']}")

        if not eval_result["passed"]:
            console.print(f"\nWide deviation detected in {display_name}.")
            console.print(
                "Form breakdown compromises safety and reinforces poor movement patterns. Rest, recover, and try again later.")
            save_results(session_log)
            raise typer.Exit()

        console.print(f"Passed {display_name}.")

    console.print("\nAll metrics passed! Excellent session.")
    save_results(session_log)


if __name__ == "__main__":
    app()