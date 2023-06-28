import os
import pyaudio
import wave
from keras.models import load_model
import numpy as np
import librosa
from threading import Thread
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import time
import whisper

# Face Emotion Detector
class FaceEmotionDetector:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(r'/home/tv-python-dev/Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
        self.classifier = load_model(r'/home/tv-python-dev/Emotion_Detection_CNN/model.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def detect_emotions(self, frame):
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
                labels.append(label)
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                labels.append('No Faces')
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, labels

# Speech Emotion Recognizer
class SpeechEmotionRecognizer:
    def __init__(self, model_path, output_folder):
        self.model = load_model(model_path)
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.RECORD_SECONDS = 60
        self.output_folder = output_folder
        self.p = pyaudio.PyAudio()

    def create_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_audio(self, video_file):
        video = mp.VideoFileClip(video_file)
        audio_file = os.path.join(self.output_folder, "audio.wav")
        video.audio.write_audiofile(audio_file)

        # Incremental file name for multiple recordings
        file_count = len(os.listdir(self.output_folder))
        new_audio_file = os.path.join(self.output_folder, f"audio_{file_count}.wav")
        os.rename(audio_file, new_audio_file)
        return new_audio_file

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


# Video Emotion Analysis
def analyze_video():
    cap = cv2.VideoCapture('/home/tv-python-dev/Downloads/treeshouse.mp4')  # Replace 'path_to_video_file' with the path to your video file
    face_emotion_detector = FaceEmotionDetector()
    recognizer = SpeechEmotionRecognizer(model_path, output_folder)
    recognizer.create_output_folder()
    audio_extracted = False

    # Initialize audio conversion
    r = sr.Recognizer()

    video_emotion_labels = []  # Store emotion labels for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for smoother processing
        resized_frame = cv2.resize(frame, (640, 480))  # Adjust the size as per your preference

        # Face Emotion Detection
        frame_with_emotions, labels = face_emotion_detector.detect_emotions(resized_frame)

        video_emotion_labels.extend(labels)

        # Display the frame with emotions
        cv2.imshow('Video Emotion Analysis', frame_with_emotions)

        # Extract audio and convert to text
        if not audio_extracted:
            audio_file = recognizer.extract_audio('/home/tv-python-dev/Downloads/treeshouse.mp4')  # Replace with the path to your video file
            recognizer.load_audio(audio_file)
            recognizer.preprocess_audio()
            predicted_emotion = recognizer.predict_emotion()
            print("Predicted Emotion:", predicted_emotion)

             # Perform speech-to-text conversion in a separate thread
            audio_thread = Thread(target=lambda: convert_audio_to_text(audio_file, r, predicted_emotion))
            audio_thread.start()

            audio_extracted = True

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save video emotion labels to a file
    output_file = 'video_emotion_labels.txt'
    with open(output_file, 'w') as f:
        for label in video_emotion_labels:
            timestamp = time.strftime("%H:%M:%S", time.gmtime())
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Emotion: {label}\n")
            f.write("\n")

def convert_audio_to_text(audio_file, recognizer, predicted_emotion):
    print("Converting audio to text...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)

    text = result["text"]
    print("Converted Text:", text)

    # Save converted text and predicted emotion to a file
    output_file = 'audio_text_emotion.txt'
    with open(output_file, 'w') as f:
        f.write(f"Predicted Emotion: {predicted_emotion}\n")
        f.write(f"Converted Text: {text}\n")
        f.write("\n")

        
        # Compare with original text (if available) to evaluate correctness
        original_text = "Original text here"  # Replace with the original text from the audio source
        accuracy = calculate_accuracy(text, original_text)
        with open('accuracy.txt', 'w') as f:
            f.write(f"accuracy: {accuracy}\n")
        print("Accuracy:", accuracy)
    return text

def calculate_accuracy(text, original_text):
    # Implement your accuracy calculation logic here
    # You can use metrics such as edit distance or word-level accuracy
    accuracy = 0.0
    return accuracy

# Usage example
model_path = '/home/tv-python-dev/SentiMent _Project_ML Model/speech_emotion_model.h5'
output_folder = '/home/tv-python-dev/SentiMent _Project_ML Model/Realtime_multi_Model_recordings'

# Start video analysis in a separate thread
video_thread = Thread(target=analyze_video)
video_thread.start()

# Wait for video analysis to finish
video_thread.join()
