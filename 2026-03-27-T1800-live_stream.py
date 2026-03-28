import cv2
import mediapipe as mp
import time

# Aliases
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 1. Define the callback function that will handle the results asynchronously
def handle_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # This function is called automatically whenever a pose is detected
    if result.pose_landmarks:
        # Example: Print the number of detected poses
        print(f"Detected {len(result.pose_landmarks)} pose(s) at {timestamp_ms}ms")

        # NOTE: If you want to draw the skeleton, you generally need to share
        # this 'result' data with your main thread, as drawing directly inside
        # a background callback can cause UI thread crashes in OpenCV.


def main():
    # 2. Configure the options, ensuring you pass the callback
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./pose_landmarker_heavy.task'),  # Ensure this file is downloaded!
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )

    # 3. Initialize the landmarker and the video capture
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return

        print("Press 'q' to quit.")

        # Track start time to generate monotonically increasing timestamps
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Convert OpenCV's BGR format to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the numpy array to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Calculate the timestamp in milliseconds
            timestamp_ms = int((time.time() - start_time) * 1000)

            # 4. Send the image to the landmarker
            # It will process in the background and trigger handle_result() when done
            landmarker.detect_async(mp_image, timestamp_ms)

            # Display the raw frame (drawing would require synchronizing the callback data)
            cv2.imshow('MediaPipe Live Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()