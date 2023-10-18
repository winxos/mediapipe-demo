import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    paint = []
    cont  = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec =  mp_drawing.DrawingSpec(color=(0,255,55),thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec =  mp_drawing.DrawingSpec(color=(0,255,0))
            )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec =  mp_drawing.DrawingSpec(color=(255,0,255))
            )

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.right_hand_landmarks:
            thumb_tip = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = math.sqrt((image.shape[1]*(thumb_tip.x - index_tip.x))**2 + (image.shape[0]*(thumb_tip.y - index_tip.y))**2)
            print(f"{distance}")
            if distance <20 and results.left_hand_landmarks:
                pt = results.left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cont.append([pt.x*image.shape[1],pt.y*image.shape[0]])
                if len(cont)>2:
                    pts = np.array(cont,np.int)
                    pts = pts.reshape(-1,1,2)
                    cv2.polylines(image,[pts],False,color=(200,50,0),thickness = 2)
            else:
                if len(cont) > 10:
                    pts = np.array(cont,np.int)
                    pts = pts.reshape(-1,1,2)
                    paint.append(pts)
                cont = []
        if len(paint) > 0:
           cv2.polylines(image,paint,False,color=(200,200,0),thickness = 3)

        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
