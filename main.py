import cv2
import mediapipe as mp
import numpy as np
import time

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

if __name__ == '__main__':
    main()