import cv2
import mediapipe as mp
import numpy as np
import time

# LEFT (Sagittal View)
# 11 Left Shoulder
left_shoulder = 11
# 23 Left Waist
left_waist = 23
# 25 Left Knee
left_knee = 25
# 27 Left Ankle
left_ankle = 27

# RIGHT
# 12 Right Shoulder
right_shoulder = 12
# 24 Right Waist
right_waist = 24
# 26 Right Knee
right_knee = 26
# 28 Right Ankle
right_ankle = 28

def calc_angle_2d(a,b,c):
    angle_ba = np.array([a.x - b.x, a.y - b.y])
    angle_bc = np.array([c.x - b.x, c.y - b.y])
    length_ba = np.linalg.norm(angle_ba)
    length_bc = np.linalg.norm(angle_bc)
    cosine_angle = np.dot(angle_ba, angle_bc) / (length_ba * length_bc)

    # Constraint the angle to
    bounded_angle = np.clip(cosine_angle, 0.0, 1.0)

    # Convert decimal to radians
    rads = np.arccos(bounded_angle)

    # Convert radians to degrees
    degrees = np.degrees(rads)

    return degrees

def calc_angle_3d(a,b,c):
    # Convert points to numpy arrays
    a, b, c = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z]), np.array([c.x, c.y, c.z])

    # Convert numpy arrays to 2d vectors
    ba, bc = a - b, c - b

    length_ba = np.linalg.norm(ba)
    length_bc = np.linalg.norm(bc)

    cosine_angle = np.dot(ba, bc) / (length_ba * length_bc)

count_state_standing = "STANDING"

def draw_skeleton(image, landmarks):
    h,w, _ = image.shape

    shoulders = (left_shoulder, right_shoulder)
    hips = (left_waist, right_waist)
    right_torso = (right_shoulder, right_waist)
    right_femur = (right_waist, right_knee)
    right_shin = (right_knee, right_ankle)
    # TODO: ALLOW TOGGLE TO LEFT SIDE?
    connections = [shoulders, hips, right_torso, right_femur, right_shin]

    # DRAW LINES
    for start_index, end_index in connections:
        start = landmarks[start_index]
        end = landmarks[end_index]

        # Style
        color = (245, 117, 66)
        thickness = 5
        radius = 5

        # Convert to pixel coords
        start_x = int(start.x * w)
        start_y = int(start.y * h)
        pt1 = (start_x, start_y)
        end_x = int(end.x * w)
        end_y = int(end.y * h)
        pt2 = (end_x, end_y)

        cv2.line(image, pt1, pt2, color, thickness)

        # Draw Joints
        for joint in [right_shoulder, right_waist, right_knee, right_ankle]:
            landmark = landmarks[joint]
            landmark_x = int(landmark.x * w)
            landmark_y = int(landmark.y * h)
            cv2.circle(image, (landmark_x, landmark_y), radius, color, thickness)

VIDEO_PATH = './anterior trim.mov'

def main():
    BaseOptions = mp.tasks.BaseOptions
    MODEL_PATH = './pose_landmarker_full.task'

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

    # TODO: Switch from video to live-stream based on progress

    with PoseLandmarker.create_from_options(PoseLandmarkerOptions) as landmarker:
        cap = cv2.VideoCapture(VIDEO_PATH)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Get timestamp
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # OpenCV to MP
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Output rendering
            output_image = frame.copy()

            # Draw skeleton if joints are found
            if detection_result.pose_landmarks:
                # Find the first person, if multiple people are detected
                landmarks = detection_result.pose_landmarks[0]


                # TODO: Is there a way to detect what angle or side the script is looking at?

                l_shoulder_landmark = landmarks[left_shoulder]
                l_waist_landmark = landmarks[left_waist]
                l_knee_landmark = landmarks[left_knee]
                l_ankle_landmark = landmarks[left_ankle]

                knee_angle = calc_angle_3d(l_waist_landmark, l_knee_landmark, l_ankle_landmark)

                # TODO: Add other KC's here

                draw_skeleton(output_image, landmarks)

            cv2.imshow("Output", output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    main()