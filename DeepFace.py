import cv2
from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Create an empty DataFrame to store the emotion data
vid_df = pd.DataFrame()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = webcam.read()

    try:
        # Perform emotion detection on the frame
        processing_data = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

        # Convert the analyzed information into a DataFrame
        frame_df = pd.DataFrame(processing_data)
        vid_df = vid_df._append(frame_df, ignore_index=True)

        # Display the frame with emotion values
        for data in processing_data:
            emotions = data['emotion']
            emotion_str = max(emotions, key=emotions.get)
            cv2.putText(frame, emotion_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except ValueError:
        pass

    # Display the resulting frame
    cv2.imshow('Webcam Emotion Detection', frame)

    # Press 'q' to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()

# Plotting the emotions against time in the video
pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()

# Calculate the sum of emotions for the entire video
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotions_values = vid_df[emotions].sum()

# Create a DataFrame to store the emotion values
score_comparisons = pd.DataFrame({'Human Emotions': emotions, 'Emotion Value from the Video': emotions_values})
score_comparisons
