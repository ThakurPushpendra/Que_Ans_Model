import pyaudio                          # Library for audio recording  
import wave           
from keras.models import load_model     # Library for model loading
import numpy as np
import librosa                          # Library for audio processing 

model = load_model('/home/tv-python-dev/SentiMent _Project_ML Model/speech_emotion_model.h5')

CHUNK = 1024                # he number of audio frames per buffer
FORMAT = pyaudio.paInt16    # Audio Dataformat
CHANNELS = 2                # the number of audio channels
RATE = 44100                #  sample rate
RECORD_SECONDS = 2          # duration of the recording
WAVE_OUTPUT_FILENAME = "output.wav" # filename to save the recorded audio.

p = pyaudio.PyAudio()  #An instance of the PyAudio class is created to handle audio-related operations.


"""The audio stream is opened with the specified format,
 channels, sample rate, and frames per buffer. This prepares the stream for recording."""


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")


"""A loop is executed to read audio data from the stream in chunks. 
Each chunk is appended to the frames list."""

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

"""The audio stream is stopped, closed, 
and the PyAudio instance is terminated to release system resources related to audio recording."""

stream.stop_stream()
stream.close()
p.terminate()

"""A wave file is opened in write mode,
 and the audio frames stored in frames are written to the file. The file is then closed."""

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


"""The recorded audio file is loaded using librosa. The load function returns the audio signal 
signal and the sample rate sr.mono=True specifies that the audio should be converted to mono 
by averaging across channels."""

# Load the recorded audio and perform emotion recognition
signal, sr = librosa.load(WAVE_OUTPUT_FILENAME, sr=RATE, mono=True)

# Reshape and normalize the audio signal
signal = librosa.util.normalize(signal)


"""The audio signal is normalized to ensure its values range between -1 and 1.
 This step helps maintain consistent amplitude levels."""
# Pad or truncate the audio signal to match the desired length

MAX_LENGTH = 40
if len(signal) < MAX_LENGTH:
    padded_signal = np.pad(signal, (0, MAX_LENGTH - len(signal)), mode='constant')
else:
    padded_signal = signal[:MAX_LENGTH]

# Reshape the signal for input to the model
signal = np.reshape(padded_signal, (1, -1))

# Perform emotion recognition
emotion_probabilities = model.predict(signal)
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
predicted_emotion_index = np.argmax(emotion_probabilities)
predicted_emotion = emotion_labels[predicted_emotion_index]

print("Predicted Emotion:", predicted_emotion)

