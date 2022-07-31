import glob, os
import cv2
import math
import mediapipe as mp
import numpy as np
import pandas as pd
from alive_progress import alive_bar

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

pose_set = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
              'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY',
              'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
              'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

right_hand = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
               'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP',
               'RING_FINGER_TIP', 'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

left_hand = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2',
                 'INDEX_FINGER_PIP2', 'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
                 'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2',
                 'RING_FINGER_DIP2', 'RING_FINGER_TIP2', 'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2',
                 'PINKY_TIP2']

for file in glob.glob("./assets/*.*"):
    # load videos from ./assets and get dimensions
    cap = cv2.VideoCapture(file)
    width = math.floor(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = math.floor(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_filename = './output/' + os.path.basename(file)
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS),
                          (width, height))

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        count = 0
        keypoints = []
        with alive_bar(total_frames, dual_line=True) as bar:
            while cap.isOpened():
                ret, frame = cap.read()
                count += 1
                # Recolor Feed
                if frame is not None:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break
                output_image = np.zeros((height, width, 3), np.uint8)

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(output_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                          )

                # 2. Right hand
                mp_drawing.draw_landmarks(output_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                          )

                # 3. Left Hand
                mp_drawing.draw_landmarks(output_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                if results.pose_world_landmarks:
                    data_main = {}
                    for i in range(len(pose_set)):
                        results.pose_world_landmarks.landmark[i].x = results.pose_world_landmarks.landmark[i].x * image.shape[0]
                        results.pose_world_landmarks.landmark[i].y = results.pose_world_landmarks.landmark[i].y * image.shape[1]
                        data_main.update(
                            {pose_set[i]: results.pose_world_landmarks.landmark[i]}
                        )
                    keypoints.append(data_main)

                if results.right_hand_landmarks:
                    data_right_hand = {}
                    for i in range(len(right_hand)):
                        results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x *\
                                                                     image.shape[0]
                        results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y *\
                                                                     image.shape[1]
                        data_main.update(
                            {right_hand[i]: results.right_hand_landmarks.landmark[i]}
                        )
                    keypoints.append(data_main)

                if results.left_hand_landmarks:
                    data_left_hand = {}
                    for i in range(len(left_hand)):
                        results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x *\
                                                                    image.shape[0]
                        results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y *\
                                                                    image.shape[1]
                        data_main.update(
                            {left_hand[i]: results.left_hand_landmarks.landmark[i]}
                        )
                    keypoints.append(data_main)

                df = pd.DataFrame(keypoints)
                df.to_csv(output_filename + '.csv')
                out.write(output_image)
                bar()

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()

cv2.destroyAllWindows()