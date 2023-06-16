import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the pre-trained FER model
model = load_model('path/to/fer_model.h5')

# Define the emotions labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Create an empty DataFrame to store the emotion data
vid_df = pd.DataFrame(columns=['emotion'])

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = webcam.read()

    # Preprocess the frame for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))

    # Perform emotion detection on the frame
    predictions = model.predict(reshaped)
    emotion_idx = np.argmax(predictions[0])
    emotion_str = emotions[emotion_idx]

    # Store the emotion in the DataFrame
    frame_data = {'emotion': emotion_str}
    vid_df = vid_df.append(frame_data, ignore_index=True)

    # Display the frame with emotion values
    cv2.putText(frame, emotion_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Emotion Detection', frame)

    # Press 'q' to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()

# Plotting the emotions against time in the video
plt.figure(figsize=(20, 8))
plt.plot(vid_df.index, pd.factorize(vid_df['emotion'])[0])
plt.xticks(vid_df.index)
plt.xlabel('Frame')
plt.ylabel('Emotion')
plt.show()

# Calculate the sum of emotions for the entire video
emotions_values = vid_df['emotion'].value_counts()

# Create a DataFrame to store the emotion values
score_comparisons = pd.DataFrame({'Human Emotions': emotions, 'Emotion Value from the Video': emotions_values})
score_comparisons
