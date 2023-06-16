import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the data from the Excel file
data = pd.read_excel('/home/tv-python-dev/Documents/DemoQA.xlsx')
questions = data['Question'].values
answers = data[['Answer1', 'Answer2', 'Answer3', 'Answer4']].values.transpose()

# Step 2: Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
questions_sequences = tokenizer.texts_to_sequences(questions)
questions_padded = pad_sequences(questions_sequences, padding='post')
num_answers = answers.shape[1]

# Step 3: Convert the target values to one-hot encoded vectors
label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers.ravel())
one_hot_answers = to_categorical(encoded_answers).reshape(-1, num_answers)

# Step 4: Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 100, input_length=questions_padded.shape[1]),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_answers, activation='softmax')
])

# Step 5: Train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(questions_padded, one_hot_answers, epochs=10, batch_size=16)

# Step 6: Evaluate the model
def get_accuracy(user_answer, stored_answers):
    user_answer_sequence = tokenizer.texts_to_sequences([user_answer])
    user_answer_padded = pad_sequences(user_answer_sequence, padding='post', maxlen=questions_padded.shape[1])
    predictions = model.predict(user_answer_padded)[0]
    max_index = predictions.argmax()
    matched_answer = stored_answers[max_index]
    accuracy = predictions[max_index]
    return matched_answer, accuracy

# Example usage:
user_question = "What is the capital of France?"
user_answer = "Paris"
matched_answer, accuracy = get_accuracy(user_answer, answers[0])

print("User Question:", user_question)
print("User Answer:", user_answer)
print("Matched Answer:", matched_answer)
print("Accuracy:", accuracy)
