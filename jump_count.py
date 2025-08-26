import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

jump_count = 0
jumping = False
threshold = 40  # adjust based on camera angle
cooldown = 0
last_jump_time = time.time()
start_time = time.time()
duration = 60  # set your timer (seconds)

y_positions = []

print("Jump counter started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get Y position of ankle or hip
        left_ankle_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
        y_positions.append(left_ankle_y)

        if len(y_positions) > 5:
            diff = y_positions[-1] - y_positions[-5]

            current_time = time.time()
            if diff < -threshold and (current_time - last_jump_time > 0.5):  # going up
                jump_count += 1
                last_jump_time = current_time

    # Countdown timer
    elapsed = time.time() - start_time
    remaining = max(0, int(duration - elapsed))

    cv2.putText(frame, f'Jumps: {jump_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.putText(frame, f'Time Left: {remaining}s', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

    cv2.imshow("Jump Counter", frame)

    if cv2.waitKey(10) & 0xFF == ord('q') or remaining <= 0:
        break

cap.release()
cv2.destroyAllWindows()
