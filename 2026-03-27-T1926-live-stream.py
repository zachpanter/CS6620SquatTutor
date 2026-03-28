import cv2
import mediapipe as mp
import numpy as np
import time
import queue

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
    cosine_angle = np.dot(angle_ba, angle_bc) / (length_ba * length_bc)
    bounded_angle = np.clip(cosine_angle, 0.0, 1.0)
    rads = np.arccos(bounded_angle)
    return np.degrees(rads)


def calc_angle_3d(a, b, c):
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    length_ba = np.linalg.norm(ba)
    length_bc = np.linalg.norm(bc)
    cosine_angle = np.dot(ba, bc) / (length_ba * length_bc)
    bounded_angle = np.clip(cosine_angle, 0.0, 1.0)
    radian_angle = np.arccos(bounded_angle)
    return np.degrees(radian_angle)


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
    MODEL_PATH = './pose_landmarker_full.task'  # Ensure this file exists
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode

    # 1. State Management for Synchronization
    rendered_frames_queue = queue.Queue(maxsize=3)
    pending_frames = {}

    # 2. Asynchronous Callback
    def handle_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # Retrieve the original frame using the exact timestamp
        if timestamp_ms not in pending_frames:
            return
        frame_to_draw = pending_frames.pop(timestamp_ms)

        # Draw skeleton if joints are found
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            l_shoulder_landmark = landmarks[left_shoulder]
            l_waist_landmark = landmarks[left_waist]
            l_knee_landmark = landmarks[left_knee]
            l_ankle_landmark = landmarks[left_ankle]

            knee_angle = calc_angle_3d(l_waist_landmark, l_knee_landmark, l_ankle_landmark)

            draw_skeleton(frame_to_draw, landmarks)

            green = (0, 255, 0)
            red = (0, 0, 255)
            if knee_angle is not None:
                color_knee = green if knee_angle < 100 else red
                cv2.putText(frame_to_draw, f"Knee Deg: {int(knee_angle)}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color_knee, 2, cv2.LINE_AA)

        # Push the rendered frame to the UI queue. If full, drop it to prevent visual lag.
        if not rendered_frames_queue.full():
            rendered_frames_queue.put(frame_to_draw)

    # 3. Configure for Live Stream
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Changed to 0 for default webcam, or replace with valid camera index
        cap = cv2.VideoCapture(0)

        # Live stream requires monotonically increasing timestamps
        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Generate current timestamp in ms
            timestamp_ms = int((time.time() - start_time) * 1000)

            # Save an unmodified copy of the frame to our buffer
            pending_frames[timestamp_ms] = frame.copy()

            # Fix: Properly assign the converted RGB frame before passing to MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send to background thread for processing
            landmarker.detect_async(mp_image, timestamp_ms)

            # Memory Management: Clear old frames from the buffer if they were skipped/dropped
            keys_to_delete = [ts for ts in pending_frames.keys() if ts < timestamp_ms - 1000]
            for k in keys_to_delete:
                del pending_frames[k]

            # Try to grab a rendered frame from the queue to display
            try:
                display_frame = rendered_frames_queue.get_nowait()
                cv2.imshow("Output", display_frame)
            except queue.Empty:
                # If inference is still working, pass and keep capturing new frames
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()