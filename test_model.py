import cv2
import mediapipe as mp
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import warnings
from playsound import playsound

warnings.filterwarnings("ignore")

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
#                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
#                23: 'X', 24: 'Y', 25: 'Z'}
labels_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 9: 'J',
               10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'H', 9: 'J', 10: 'K', 11: 'L',
#                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R'}
cap = cv2.VideoCapture(0)


def mediapipe_detection(image1, model1):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1.flags.writeable = False
    results1 = model1.process(image1)
    image1.flags.writeable = True
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    return image1, results1


def draw_landmarks(image1, results1):
    # mp_drawing.draw_landmarks(image1, results1.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
    #                           mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
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


# while True:
#     ret,frame=cap.read()
#     # frame = cv2.imread('./data/6/G (5).jpg')
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#     data_aux = []
#     x_ = []
#     y_ = []
#     H, W, _ = frame.shape
#     if results.multi_hand_landmarks and len(results.multi_hand_landmarks) ==2:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x)
#                 data_aux.append(y)
#                 x_.append(x)
#                 y_.append(y)
#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10
counter = 1
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        # frame = cv2.imread('./data/B/B (5).jpg')
        counter += 1
        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_landmarks(image, results)
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        predict = model.predict([keypoints])
        # predicted_character = labels_dict[predict[0]]
        # if counter % 10 == 0:
        #     playsound('sounds/{}.mp3'.format(predict[0].lower()))
        print(predict[0])
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(predict[0]), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # predict = model.predict([np.asarray(data_aux)])
        # predicted_character = labels_dict[int(predict[0])]
        # print(predicted_character)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Frame', image)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
