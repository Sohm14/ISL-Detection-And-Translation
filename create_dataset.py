import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import matplotlib.pyplot as plt

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,max_num_hands=2)
#
# Data_dir = './data'
# data = []
# labels = []
# for dir_ in os.listdir(Data_dir):
#     for img_path in os.listdir(os.path.join(Data_dir, dir_)):
#         data_aux = []
#         img = cv2.imread(os.path.join(Data_dir, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             detected_hand_count = len(results.multi_hand_landmarks)
#             if detected_hand_count > 0:
#                 hand_landmarks_1 = results.multi_hand_landmarks[0]  # First detected hand
#                 hand_landmarks_2 = None
#
#                 # Check if a second hand is detected
#                 if detected_hand_count > 1:
#                     hand_landmarks_2 = results.multi_hand_landmarks[1]  # Second detected hand
#
#                 for i in range(len(hand_landmarks_1.landmark)):
#                     x1 = hand_landmarks_1.landmark[i].x
#                     y1 = hand_landmarks_1.landmark[i].y
#
#                     x2 = 0.0  # Default values if the second hand is not detected
#                     y2 = 0.0
#                     if hand_landmarks_2 is not None:
#                         x2 = hand_landmarks_2.landmark[i].x
#                         y2 = hand_landmarks_2.landmark[i].y
#
#                     data_aux.extend([x1, y1, x2, y2])
#
#                     data.append(data_aux)
#                     labels.append(dir_)
# hand_landmarks_1 = None
# hand_landmarks_2 = None
# for hand_landmarks in results.multi_hand_landmarks:
#     if hand_landmarks.landmark[0].x < 0.5:
#         hand_landmarks_1 = hand_landmarks
#     else:
#         hand_landmarks_2 = hand_landmarks
#
# if hand_landmarks_1 is not None and hand_landmarks_2 is not None:
#     for i in range(len(hand_landmarks_1.landmark)):
#         x1 = hand_landmarks_1.landmark[i].x
#         y1 = hand_landmarks_1.landmark[i].y
#
#         x2 = hand_landmarks_2.landmark[i].x
#         y2 = hand_landmarks_2.landmark[i].y
#
#         data_aux.extend([x1, y1, x2, y2])
#
#     data.append(data_aux)
#     labels.append(dir_)
#         data_aux.append(x)
#         data_aux.append(y)
# data.append(data_aux)
# labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()

#         plt.figure()
#         plt.imshow(img_rgb)
#
# plt.show()
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image1, model1):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1.flags.writeable = False
    results1 = model1.process(image1)
    image1.flags.writeable = True
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    return image1, results1


def draw_landmarks(image1, results1):
    mp_drawing.draw_landmarks(image1, results1.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image1, results1.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image1, results1.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results1):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results1.pose_landmarks.landmark]).flatten() if results1.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in
                     results1.face_landmarks.landmark]).flatten() if results1.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results1.left_hand_landmarks.landmark]).flatten() if results1.left_hand_landmarks else np.zeros(
        21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results1.right_hand_landmarks.landmark]).flatten() if results1.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


actions = np.array([
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
])
Data_dir = './data'
no_sequences = 16
sequence_length = 10
Data_path = os.path.join('Mp_Data1')
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(Data_path, action, str(sequence)))
        except:
            pass
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        a=0
        for sequence in range(no_sequences):
            for frame_no in range(sequence_length):
                if action != 'E':
                    img = cv2.imread('data/{}/{} ({}).jpg'.format(action, action, frame_no + a+1))

                else:
                    img = cv2.imread('data/{}/{}1 ({}).jpg'.format(action, action, frame_no + a + 1))
                cv2.imshow('img', img)
                # cv2.waitKey(0)
                image, results = mediapipe_detection(img, holistic)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(Data_path, action, str(sequence), str(frame_no))
                np.save(npy_path, keypoints)
            a = a + sequence_length
