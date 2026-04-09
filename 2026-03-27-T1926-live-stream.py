import cv2
import mediapipe as mp
import numpy as np
import time
import queue
import typer
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
        (right_shoulder, right_waist), (right_waist, right_knee), (right_knee, right_ankle)
    ]
    for start_index, end_index in connections:
        start, end = landmarks[start_index], landmarks[end_index]
        color, thickness, radius = (245, 117, 66), 5, 5
        cv2.line(image, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), color, thickness)
        for joint in [right_shoulder, right_waist, right_knee, right_ankle]:
            lm = landmarks[joint]
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), radius, color, thickness)


# --- TRACKING SESSION WRAPPER ---
def run_tracking_session(focus_metric: str = "all") -> dict:
    BaseOptions = mp.tasks.BaseOptions
    MODEL_PATH = './pose_landmarker_full.task'
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode

    rendered_frames_queue = queue.Queue(maxsize=3)
    pending_frames = {}

    # Session state
    session_data = {"knee_depth": [], "valgus": [], "back_lean": []}
    session_start = time.time()
    tracking_state = {"started": False}

    def handle_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if timestamp_ms not in pending_frames: return
        frame_to_draw = pending_frames.pop(timestamp_ms)
        h, w, _ = frame_to_draw.shape

        current_time = time.time()
        elapsed = current_time - session_start

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame_to_draw, landmarks)

            # --- PHASE 1: COUNTDOWN GRACE PERIOD ---
            if elapsed < 8.0:
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

            # --- PHASE 2: ACTIVE TRACKING ---
            else:
                if not tracking_state["started"]:
                    # Trigger the terminal chime on the first active frame
                    print('\a', end='', flush=True)
                    tracking_state["started"] = True

                metrics_to_display = []

                if focus_metric in ["all", "knee_depth"] and check_visibility(landmarks,
                                                                              [left_waist, left_knee, left_ankle]):
                    knee_angle = calc_angle_3d(landmarks[left_waist], landmarks[left_knee], landmarks[left_ankle])
                    metrics_to_display.append(f"Knee Depth: {int(knee_angle)} deg")
                    # Append tuple of (timestamp, value)
                    session_data["knee_depth"].append((current_time, knee_angle))

                if focus_metric in ["all", "valgus"] and check_visibility(landmarks, [left_waist, left_knee, left_ankle,
                                                                                      right_waist, right_knee,
                                                                                      right_ankle]):
                    l_valgus = calc_angle_2d(landmarks[left_waist], landmarks[left_knee], landmarks[left_ankle])
                    r_valgus = calc_angle_2d(landmarks[right_waist], landmarks[right_knee], landmarks[right_ankle])
                    avg_valgus = (l_valgus + r_valgus) / 2
                    metrics_to_display.append(f"Valgus (L/R): {int(l_valgus)} / {int(r_valgus)}")
                    session_data["valgus"].append((current_time, avg_valgus))

                if focus_metric in ["all", "back_lean"] and check_visibility(landmarks, [left_shoulder, left_waist]):
                    back_slant = calc_vertical_angle(landmarks[left_shoulder], landmarks[left_waist])
                    metrics_to_display.append(f"Back Lean: {int(back_slant)} deg")
                    session_data["back_lean"].append((current_time, back_slant))

                y_offset = 50
                for text in metrics_to_display:
                    cv2.putText(frame_to_draw, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    y_offset += 35

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

        # --- PHASE 3: END GRACE PERIOD DATA PURGE ---
    end_time = time.time()
    cutoff_time = end_time - 2.0

    clean_data = {}
    for metric, tuples in session_data.items():
        # Keep only values recorded before the final 8 seconds, stripping the timestamp
        clean_data[metric] = [val for ts, val in tuples if ts <= cutoff_time]

    return clean_data


# --- EVALUATION LOGIC ---
def evaluate_performance(data: dict, experience: str) -> str:
    """Returns the name of the metric that needs the most work."""
    # Default fallback if no data was captured
    if not any(data.values()):
        return "knee_depth"

    # Grab the max deviations during the set
    max_lean = max(data["back_lean"]) if data["back_lean"] else 0
    min_knee = min(data["knee_depth"]) if data["knee_depth"] else 180
    avg_valgus = sum(data["valgus"]) / len(data["valgus"]) if data["valgus"] else 180

    # Stricter tolerances for experienced lifters
    lean_threshold = 35 if experience == "experienced" else 45
    depth_threshold = 80 if experience == "experienced" else 90

    if max_lean > lean_threshold:
        return "back_lean"
    elif min_knee > depth_threshold:
        return "knee_depth"
    elif avg_valgus < 170:  # Valgus deviation (straight leg is ~180)
        return "valgus"

    return "knee_depth"  # Default to depth if everything is perfect


# --- TYPER CLI ---
@app.command()
def main():
    console.print("[bold green]Starting Biomechanics Tracker[/bold green]")

    # 1. Ask experience level
    experience = typer.prompt(
        "Are you a [beginner] or [experienced] lifter?",
        default="beginner"
    ).lower()

    # 2. Ask to begin
    typer.confirm(
        "\nReady to begin the initial tracking session? (Press Enter to start, 'q' in the window to quit)",
        abort=False
    )

    # 3. Run Session 1
    console.print("\n[blue]Running Initial Assessment...[/blue]")
    session_data = run_tracking_session(focus_metric="all")

    # 4. Evaluate
    focus_target = evaluate_performance(session_data, experience)
    console.print(f"\n[bold yellow]Evaluation Complete.[/bold yellow]")
    console.print(
        f"Based on the joint tracking, your target area for the next set is: [bold red]{focus_target.replace('_', ' ').title()}[/bold red]")

    # 5. Start Session 2
    typer.confirm(
        f"\nReady to begin the isolated set focusing purely on {focus_target}? (Press Enter to start)",
        abort=False
    )

    console.print(f"\n[blue]Running Focused Assessment: {focus_target}...[/blue]")
    run_tracking_session(focus_metric=focus_target)

    console.print("\n[bold green]Session complete. Good work.[/bold green]")


if __name__ == "__main__":
    app()