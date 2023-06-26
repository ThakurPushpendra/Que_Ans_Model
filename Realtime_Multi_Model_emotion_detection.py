import os
import pyaudio
import wave
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import librosa
from threading import Thread
import cv2

# Face Emotion Detector
class FaceEmotionDetector:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(r'/home/tv-python-dev/Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
        self.classifier = load_model(r'/home/tv-python-dev/Emotion_Detection_CNN/model.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_emotions(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = self.classifier.predict(roi)[0]
                    label = self.emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Speech Emotion Recognizer
class SpeechEmotionRecognizer:
    def __init__(self, model_path, output_folder):
        self.model = load_model(model_path)
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.RECORD_SECONDS = 10
        self.output_folder = output_folder
        self.p = pyaudio.PyAudio()

    def create_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def record_audio(self):
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)

        print("* recording")

        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        self.p.terminate()

        self.create_output_folder()
        filename = f"output_{len(os.listdir(self.output_folder)) + 1}.wav"
        wave_output_path = os.path.join(self.output_folder, filename)
        wf = wave.open(wave_output_path, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def load_audio(self, wave_file):
        self.signal, self.sr = librosa.load(wave_file, sr=self.RATE, mono=True)

    def preprocess_audio(self):
        self.signal = librosa.util.normalize(self.signal)

        MAX_LENGTH = 40
        if len(self.signal) < MAX_LENGTH:
            padded_signal = np.pad(self.signal, (0, MAX_LENGTH - len(self.signal)), mode='constant')
        else:
            padded_signal = self.signal[:MAX_LENGTH]

        self.signal = np.reshape(padded_signal, (1, -1))

    def predict_emotion(self):
        emotion_probabilities = self.model.predict(self.signal)
        emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        predicted_emotion_index = np.argmax(emotion_probabilities)
        predicted_emotion = emotion_labels[predicted_emotion_index]

        return predicted_emotion

# Usage example
def run_emotion_detection():
    face_detector = FaceEmotionDetector()
    face_detector.detect_emotions()

def run_emotion_recognition():
    model_path = '/home/tv-python-dev/SentiMent _Project_ML Model/speech_emotion_model.h5'
    output_folder = '/home/tv-python-dev/SentiMent _Project_ML Model/Realtime_multi_Model'
    recognizer = SpeechEmotionRecognizer(model_path, output_folder)

    recognizer.record_audio()
    recognizer.load_audio(os.path.join(output_folder, "output_1.wav"))  # Example for loading a specific WAV file
    recognizer.preprocess_audio()
    predicted_emotion = recognizer.predict_emotion()

    print("Predicted Emotion:", predicted_emotion)

if __name__ == '__main__':
    # Create separate threads for face emotion detection and speech emotion recognition
    face_thread = Thread(target=run_emotion_detection)
    speech_thread = Thread(target=run_emotion_recognition)

    # Start both threads
    face_thread.start()
    speech_thread.start()

    # Wait for both threads to finish
    face_thread.join()
    speech_thread.join()
