import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time 
import matplotlib.pyplot as plt

def plot_world_landmarks(
    plt,
    ax_list,
    multi_hands_landmarks,
    multi_handedness,
    visibility_th=0.5,
):
    ax_list[0].cla()
    ax_list[0].set_xlim3d(-0.1, 0.1)
    ax_list[0].set_ylim3d(-0.1, 0.1)
    ax_list[0].set_zlim3d(-0.1, 0.1)
    ax_list[1].cla()
    ax_list[1].set_xlim3d(-0.1, 0.1)
    ax_list[1].set_ylim3d(-0.1, 0.1)
    ax_list[1].set_zlim3d(-0.1, 0.1)

    for landmarks, handedness in zip(multi_hands_landmarks, multi_handedness):
        handedness_index = 0
        if handedness.classification[0].label == 'Left':
            handedness_index = 0
        elif handedness.classification[0].label == 'Right':
            handedness_index = 1

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_point.append(
                [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

        palm_list = [0, 1, 5, 9, 13, 17, 0]
        thumb_list = [1, 2, 3, 4]
        index_finger_list = [5, 6, 7, 8]
        middle_finger_list = [9, 10, 11, 12]
        ring_finger_list = [13, 14, 15, 16]
        pinky_list = [17, 18, 19, 20]

        palm_x, palm_y, palm_z = [], [], []
        for index in palm_list:
            point = landmark_point[index][1]
            palm_x.append(point[0])
            palm_y.append(point[2])
            palm_z.append(point[1] * (-1))

        thumb_x, thumb_y, thumb_z = [], [], []
        for index in thumb_list:
            point = landmark_point[index][1]
            thumb_x.append(point[0])
            thumb_y.append(point[2])
            thumb_z.append(point[1] * (-1))

        index_finger_x, index_finger_y, index_finger_z = [], [], []
        for index in index_finger_list:
            point = landmark_point[index][1]
            index_finger_x.append(point[0])
            index_finger_y.append(point[2])
            index_finger_z.append(point[1] * (-1))

        middle_finger_x, middle_finger_y, middle_finger_z = [], [], []
        for index in middle_finger_list:
            point = landmark_point[index][1]
            middle_finger_x.append(point[0])
            middle_finger_y.append(point[2])
            middle_finger_z.append(point[1] * (-1))

        ring_finger_x, ring_finger_y, ring_finger_z = [], [], []
        for index in ring_finger_list:
            point = landmark_point[index][1]
            ring_finger_x.append(point[0])
            ring_finger_y.append(point[2])
            ring_finger_z.append(point[1] * (-1))

        pinky_x, pinky_y, pinky_z = [], [], []
        for index in pinky_list:
            point = landmark_point[index][1]
            pinky_x.append(point[0])
            pinky_y.append(point[2])
            pinky_z.append(point[1] * (-1))

        ax_list[handedness_index].plot(palm_x, palm_y, palm_z)
        ax_list[handedness_index].plot(thumb_x, thumb_y, thumb_z)
        ax_list[handedness_index].plot(index_finger_x, index_finger_y,
                                       index_finger_z)
        ax_list[handedness_index].plot(middle_finger_x, middle_finger_y,
                                       middle_finger_z)
        ax_list[handedness_index].plot(ring_finger_x, ring_finger_y,
                                       ring_finger_z)
        ax_list[handedness_index].plot(pinky_x, pinky_y, pinky_z)

    plt.pause(.001)

    return

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 2,model_complexity=1)

    c_time = 0
    p_time = 0

    fig = plt.figure()
    r_ax = fig.add_subplot(121, projection="3d")
    l_ax = fig.add_subplot(122, projection="3d")
    fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    # cap = cv2.VideoCapture('rtsp://12.10.2.250:8080/h264.sdp')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.flip(image, 1)

        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for hand_landmarks, classification in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness = classification.classification[0].label
                wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1])
                wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0])

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(10, 10, 10), thickness=5, circle_radius=4), 
                                          mp_drawing.DrawingSpec(color=(250, 250, 250), thickness=2, circle_radius=2))

                cv2.putText(image, handedness, (wrist_x, wrist_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                bounding_box = [min(int(l.x * image.shape[1]), image.shape[1] - 1) for l in hand_landmarks.landmark]
                x_min, x_max = min(bounding_box), max(bounding_box)
                y_min, y_max = min(int(l.y * image.shape[0]) for l in hand_landmarks.landmark), max(int(l.y * image.shape[0]) for l in hand_landmarks.landmark)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)

                if results.multi_hand_world_landmarks is not None:
                    plot_world_landmarks(
                        plt,
                        [r_ax, l_ax],
                        results.multi_hand_world_landmarks,
                        results.multi_handedness,
                    )
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time

        cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX, 3,(255,255,255),3)

        cv2.imwrite(os.path.join('./sample_imgs/misc/', '{}.jpg'.format(int(time.time()))), image)
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
