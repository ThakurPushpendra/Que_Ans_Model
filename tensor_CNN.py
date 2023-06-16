import pandas as pd 
import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Embedding,Conv1D, GlobalMaxPooling1D, Dense
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split


# Step 1: Read data from Excel file
data = pd.read_excel('/home/tv-python-dev/Documents/DemoQA.xlsx')  # Replace 'data.xlsx' with your actual file path
questions = data["Question"].tolist()
answers = data.iloc[:, 1:].values.tolist()

# Step 2: Preprocess the data
question_tokenizer = tf.keras.preprocessing.text.Tokenizer()
question_tokenizer.fit_on_texts(questions)
num_questions = len(question_tokenizer.word_index) + 1

answer_tokenizer = tf.keras.preprocessing.text.Tokenizer()
answer_tokenizer.fit_on_texts([answer for ans_list in answers for answer in ans_list])
num_answers = len(answer_tokenizer.word_index) + 1
question_sequences = question_tokenizer.texts_to_sequences(questions)
answer_sequences = [answer_tokenizer.texts_to_sequences(ans_list) for ans_list in answers]
max_sequence_length = max(len(seq) for seq in question_sequences + [ans for ans_list in answer_sequences for ans in ans_list])
question_sequences = pad_sequences(question_sequences, maxlen=max_sequence_length)
answer_sequences = [pad_sequences(ans_list, maxlen=max_sequence_length) for ans_list in answer_sequences]

# Print shapes and sequences
print(f'Shape of question_sequences: {question_sequences.shape}')
print(f'Shape of answer_sequences: {answer_sequences[0].shape}')
print(f'Example question sequence: {question_sequences[0]}')
print(f'Example answer sequence: {answer_sequences[0][0]}')


# Step 3: Design the model architecture
embedding_dim = 100
filters = 64
kernel_size = 3
num_classes = len(answers[0])  # Number of classes is the number of columns in the answers.

model = Sequential()

model.add(Embedding(num_answers, embedding_dim, input_length=max_sequence_length))

# model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(64, activation='relu'))
print("******",model.add(Dense(64, activation='relu')))
# model.add(Dense(len(answers[0]), activation='softmax'))
model.add(Dense(num_classes, activation='softmax')) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
print("========ModelSummary",model.summary())
# Step 4: Train the model
# X = np.array(answer_sequences)
X = np.concatenate(answer_sequences, axis=0)
y = np.repeat(question_sequences, len(answer_sequences[0]), axis=0)




# y = np.repeat(question_sequences, 4, axis=0)  # Replicate question sequences
# y = np.repeat(question_sequences, len(answers[0]), axis=0)
# y_val = np.repeat(question_sequences, len(answers[0]), axis=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Number of samples in X_train: {len(X_train)}')  
print(f'Number of samples in y_train: {len(y_train)}')

# # Reshape X_train and X_val
# X_train = X_train.reshape((-1, max_sequence_length))
# X_val = X_val.reshape((-1, max_sequence_length))
# y_train = y_train.reshape((-1, max_sequence_length))
# y_val = y_val.reshape((-1, max_sequence_length))

batch_size = 16
epochs = 10
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
print(model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val)))

# Step 5: Evaluate the model
# Evaluate on test set or new question-answer pairs
# Evaluate on test set
# X_test = np.transpose(answer_sequences)  # Assuming the same data for testing
# y_test = np.transpose(question_sequences)  # Assuming the same data for testing


# loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
loss, accuracy = model.evaluate(X, y, batch_size=batch_size)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# Step 6: Deploy and use the model
# Implement the system to ask questions, get user answers, and compute similarity scores

# X_test = np.transpose(answer_sequences)
# y_test = np.transpose(question_sequences)


def ask_question(question):
    user_answer = input(question + " ")
    user_answer_sequence = pad_sequences(answer_tokenizer.texts_to_sequences([user_answer]), maxlen=max_sequence_length)
    system_answers = model.predict(user_answer_sequence)
    system_answer = answer_tokenizer.sequences_to_texts([np.argmax(system_answers)])[0]
    similarity_score = np.max(system_answers) * 100
    
    return system_answer, similarity_score

while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    system_answer, similarity_score = ask_question(question)
    print(f"System Answer: {system_answer}")
    print(f"Similarity Score: {similarity_score:.2f}%")


