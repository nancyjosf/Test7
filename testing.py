import numpy as np
import cv2
import mediapipe as mp
import torch
from ai import ExerciseClassifier

class ExerciseClassifierApp:
    def __init__(self, model_path: str, label_encoder_path: str, input_size: int, num_classes: int):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = self.load_model(model_path)
        self.label_encoder = self.load_label_encoder(label_encoder_path)
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.frame_count = 0
        self.correct_predictions = 0

    def load_model(self, model_path: str):
        model = ExerciseClassifier(self.input_size, self.num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def load_label_encoder(self, label_encoder_path: str):
        classes = np.load(label_encoder_path, allow_pickle=True)
        return classes

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_pose.process(image_rgb)
        if result.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
            return landmarks
        else:
            return None

    def classify_frame(self, frame):
        landmarks = self.process_frame(frame)
        if landmarks is None:
            return "No valid pose detected", False

        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            _, prediction = torch.max(outputs, 1)
            class_name = self.label_encoder[prediction.item()]
            return class_name, True

    def display_frame(self, frame, classification_result, accuracy):
        cv2.putText(frame, f'Class: {classification_result}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Exercise Classification', frame)

    def run(self, frame_limit=500):
        cap = cv2.VideoCapture(0)
        while self.frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break

            classification_result, is_valid = self.classify_frame(frame)
            print(f'Classification Result: {classification_result}')

            if is_valid:
                self.correct_predictions += 1

            accuracy = self.correct_predictions / (self.frame_count + 1)
            self.display_frame(frame, classification_result, accuracy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    input_size = 66  # 33 landmarks * 2 (x, y)
    num_classes = 4  # Adjust based on your dataset

    model_path = 'exercise_classifier.pth'
    label_encoder_path = 'label_encoder.npy'

    app = ExerciseClassifierApp(model_path, label_encoder_path, input_size, num_classes)
    app.run()
