import mediapipe as mp
import numpy as np
import cv2, time, keyboard, yaml

cap = cv2.VideoCapture(0)

def record(mtx, dist, images):
    pTime = 0

    while True:
        _, img1 = cap.read()
        img2 = cv2.undistort(img1, mtx, dist)

        images.append(img2)
        if len(images) > 1: images.pop(0)

        # Measure FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img2, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 0), 2)
        cv2.imshow('RECORD', img2)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def detect(images):
    ticks = 0
    data = []
    begin_time = time.time()
    hands = mp.solutions.hands.Hands(max_num_hands=1)

    while True:
        if len(images) > 0:
            img = cv2.cvtColor(np.copy(images[0]), cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            frame = []
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for idx, lm in enumerate(handLms.landmark):
                        if idx in [4, 8, 12, 16, 20]:
                            frame.append([idx, lm.x, lm.y, lm.z])
            data.append(frame)
            ticks += 1
        
        # Record result
        if keyboard.is_pressed('escape'):
            fps = ticks / (time.time() - begin_time)
            print(f'{fps} frames per second average')
            with open('result.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            break
