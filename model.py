

import cv2
import mediapipe as mp
import os

class PullUpCounter:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.count = 0
        self.position = "up"
        self.said = False
        self.status = 'false'
        self.rgb = (255, 0, 0)

    def start_detection(self, frame):
        with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imlist = []

            if result.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(result.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    X, Y = int(lm.x * w), int(lm.y * h)
                    imlist.append([id, X, Y])

                if len(imlist) != 0:
                    if imlist[12][2] >= imlist[13][2] and imlist[11][2] >= imlist[14][2]:
                        self.position = "down"
                        self.status = "true"
                        if not self.said:
                            os.system('say -v "karen" "down"')
                            self.said = True
                    elif imlist[12][2] <= imlist[13][2] and imlist[11][2] <= imlist[14][2] and self.position == "down":
                        self.position = "up"
                        self.count += 1
                        self.status = "false"
                        print("Pull_Up completed:", self.count)
                        os.system('say -v "karen" "%s"' % self.count)

            self.display_count(image)
            return image

    def display_count(self, image):
        posit = (int(image.shape[1] / 3 - 268 / 2), int(image.shape[0] / 4 - 36 / 2))
        cv2.putText(image, 'Pull_Up completed: %s' % self.count, posit, 0, 1, (255, 0, 0), 3, 15)
        if self.status == 'false':
            self.rgb = (0, 0, 255)
        else:
            self.rgb = (255, 0, 0)

        cv2.putText(image, 'status: %s' % self.status, (50, 50), 0, 1, self.rgb, 3, 15)
        return image

