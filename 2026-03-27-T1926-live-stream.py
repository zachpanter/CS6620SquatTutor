import cv2
import mediapipe as mp
import numpy as np
import time
import queue

# PPM
left_ear = 7
right_ear = 8
KNOWN_EAR_DIST_CM = 20.0

# LEFT (Sagittal View)
left_shoulder = 11
left_waist = 23
left_knee = 25
left_ankle = 27

# RIGHT
right_shoulder = 12
right_waist = 24
right_knee = 26
right_ankle = 28


def calc_angle_2d(a, b, c):
    angle_ba = np.array([a.x - b.x, a.y - b.y])
    angle_bc = np.array([c.x - b.x, c.y - b.y])
    length_ba = np.linalg.norm(angle_ba)
    length_bc = np.linalg.norm(angle_bc)

    # Handle edge case where points perfectly overlap
    if length_ba == 0 or length_bc == 0:
        return 0.0

    cosine_angle = np.dot(angle_ba, angle_bc) / (length_ba * length_bc)
    bounded_angle = np.clip(cosine_angle, -1.0, 1.0)
    rads = np.arccos(bounded_angle)
    return np.degrees(rads)


def calc_angle_3d(a, b, c):
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    length_ba = np.linalg.norm(ba)
    length_bc = np.linalg.norm(bc)

    if length_ba == 0 or length_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (length_ba * length_bc)
    bounded_angle = np.clip(cosine_angle, -1.0, 1.0)
    radian_angle = np.arccos(bounded_angle)
    return np.degrees(radian_angle)


def calc_vertical_angle(top, bottom):
    # Calculates the angle of the torso relative to true vertical (gravity)
    vec = np.array([top.x - bottom.x, top.y - bottom.y])
    vert = np.array([0, -1])  # Vector pointing straight up

    length_vec = np.linalg.norm(vec)
    if length_vec == 0:
        return 0.0

    cosine_angle = np.dot(vec, vert) / (length_vec * np.linalg.norm(vert))
    bounded_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(bounded_angle))


def check_visibility(landmarks, indices, threshold=0.6):
    # Returns True only if ALL requested landmarks meet the confidence threshold
    for idx in indices:
        if landmarks[idx].visibility < threshold:
            return False
    return True


def draw_skeleton(image, landmarks):
    h, w, _ = image.shape
    shoulders = (left_shoulder, right_shoulder)
    hips = (left_waist, right_waist)
    right_torso = (right_shoulder, right_waist)
    right_femur = (right_waist, right_knee)
    right_shin = (right_knee, right_ankle)
    connections = [shoulders, hips, right_torso, right_femur, right_shin]

    for start_index, end_index in connections:
        start = landmarks[start_index]
        end = landmarks[end_index]
        color = (245, 117, 66)
        thickness = 5
        radius = 5

        start_x, start_y = int(start.x * w), int(start.y * h)
        end_x, end_y = int(end.x * w), int(end.y * h)
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

        for joint in [right_shoulder, right_waist, right_knee, right_ankle]:
            landmark = landmarks[joint]
            landmark_x, landmark_y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (landmark_x, landmark_y), radius, color, thickness)


def main():
    BaseOptions = mp.tasks.BaseOptions
    MODEL_PATH = './pose_landmarker_full.task'
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode

    # State Management for Synchronization
    rendered_frames_queue = queue.Queue(maxsize=3)
    pending_frames = {}



    # Asynchronous Callback
    def handle_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if timestamp_ms not in pending_frames:
            return
        frame_to_draw = pending_frames.pop(timestamp_ms)
        h, w, _ = frame_to_draw.shape

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame_to_draw, landmarks)

            # Accumulate readouts dynamically
            metrics_to_display = []

            # 1. Knee Angle (3D Sagittal) - Using Left Side
            if check_visibility(landmarks, [left_waist, left_knee, left_ankle]):
                knee_angle = calc_angle_3d(landmarks[left_waist], landmarks[left_knee], landmarks[left_ankle])
                metrics_to_display.append(f"Knee Depth: {int(knee_angle)} deg")

            # 2. Knee Valgus (2D Frontal Plane)
            if check_visibility(landmarks, [left_waist, left_knee, left_ankle, right_waist, right_knee, right_ankle]):
                # Calculates 2D angle of Hip->Knee->Ankle. Deviation indicates valgus/varus.
                l_valgus = calc_angle_2d(landmarks[left_waist], landmarks[left_knee], landmarks[left_ankle])
                r_valgus = calc_angle_2d(landmarks[right_waist], landmarks[right_knee], landmarks[right_ankle])
                metrics_to_display.append(f"Valgus (L/R): {int(l_valgus)} / {int(r_valgus)}")

            # 3. Back Slant (Torso vs Vertical)
            if check_visibility(landmarks, [left_shoulder, left_waist]):
                back_slant = calc_vertical_angle(landmarks[left_shoulder], landmarks[left_waist])
                metrics_to_display.append(f"Back Lean: {int(back_slant)} deg")

            # 4. Stance Width (Ankle-to-Ankle in pixels)
            if check_visibility(landmarks, [left_ankle, right_ankle]):
                l_ank = np.array([landmarks[left_ankle].x * w, landmarks[left_ankle].y * h])
                r_ank = np.array([landmarks[right_ankle].x * w, landmarks[right_ankle].y * h])
                stance_width = np.linalg.norm(l_ank - r_ank)
                metrics_to_display.append(f"Stance Width: {int(stance_width)} px")

            # Render text block
            y_offset = 50
            for text in metrics_to_display:
                cv2.putText(frame_to_draw, text, (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 35

        if not rendered_frames_queue.full():
            rendered_frames_queue.put(frame_to_draw)

    # Configure for Live Stream
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            pending_frames[timestamp_ms] = frame.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            landmarker.detect_async(mp_image, timestamp_ms)

            # Memory cleanup
            keys_to_delete = [ts for ts in pending_frames.keys() if ts < timestamp_ms - 1000]
            for k in keys_to_delete:
                del pending_frames[k]

            try:
                display_frame = rendered_frames_queue.get_nowait()
                cv2.imshow("Biomechanics Dashboard", display_frame)
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()