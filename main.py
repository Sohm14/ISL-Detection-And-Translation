import os
import cv2
import time
Data_dir = './data'
if not os.path.exists(Data_dir):
    os.makedirs(Data_dir)

number_of_classes = 26
dataset_size = 200
cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(Data_dir, str(j))):
        os.makedirs(os.path.join(Data_dir, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready,press Q', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) == ord('q'):
            time.sleep(3)
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(Data_dir, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
