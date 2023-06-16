import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import random

# Read the dataset from an Excel file
def read_dataset(file_path):
    file_path = '/home/tv-python-dev/Documents/DemoQA.xlsx'
    df = pd.read_excel(file_path)
    questions = df.iloc[:, 0].tolist()
    answers = df.iloc[:, 1:].values.tolist()
    return questions, answers

# Preprocess the dataset
def preprocess_dataset(questions, answers):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []
    labels = []
    
    for i, question in enumerate(questions):
        for answer in answers[i]:
            encoded = tokenizer.encode_plus(
                question, answer,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(1)  # All stored answers are considered correct
    
    return (
        tf.convert_to_tensor(input_ids),
        tf.convert_to_tensor(attention_masks),
        tf.convert_to_tensor(labels)
    )

# Define the model architecture
def build_model():
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32)
    
    pooled_output = bert(input_ids, attention_mask=attention_mask)[1]

    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)
    
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model

# Train the model
def train_model(model, input_ids, attention_masks, labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    model.fit(
        x=[input_ids, attention_masks],
        y=labels,
        batch_size=8,
        epochs=3,
        validation_split=0.2
    )

# Evaluate the user's answer
def evaluate_answer(model, question, user_answer):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded = tokenizer.encode_plus(
        question, user_answer,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids = tf.convert_to_tensor([encoded['input_ids']])
    attention_mask = tf.convert_to_tensor([encoded['attention_mask']])
    
    prediction = model.predict([input_ids, attention_mask])[0][0]
    accuracy_percentage = round(prediction * 100, 2)
    return accuracy_percentage

# Save and reset the model
def save_model(model, save_path):
    model.save_weights(save_path)

def reset_model(model, save_path):
    model.load_weights(save_path)

# Example usage
file_path = '/home/tv-python-dev/Documents/DemoQA.xlsx'
questions, answers = read_dataset(file_path)
input_ids, attention_masks, labels = preprocess_dataset(questions, answers)
model = build_model()
train_model(model, input_ids, attention_masks, labels)

# Save the model
save_path = 'bert_model_weights.h5'
save_model(model, save_path)

# Reset the model
reset_model(model, save_path)

# Shuffle the questions
random.shuffle(questions)


# # Collect user answers
# user_answers = []
# for question in questions:
#     user_answer = input(f"Question: {question}\nYour Answer: ")
#     user_answers.append(user_answer)

# # Calculate overall accuracy
# total_accuracy = 0
# for i, question in enumerate(questions):
#     accuracy = evaluate_answer(model, question, user_answers[i])
#     total_accuracy += accuracy

# if len(questions) > 0:
#     overall_accuracy = total_accuracy / len(questions)
#     print(f"\nOverall accuracy of the model's evaluation: {overall_accuracy}")
# else:
#     print("No questions were answered.")

# Ask questions and evaluate user answers
total_questions = 0
total_accuracy = 0


while True:
    # Select a random question from the dataset
    random_index = random.randint(0, len(questions) - 1)
    question = questions[random_index]

    # Ask the user the selected question
    user_answer = input(f"Question: {question}\nYour Answer: ")

    # Evaluate the user's answer
    accuracy = evaluate_answer(model, question, user_answer)
    print(f"The accuracy of your answer is: {accuracy}")

    # Update the total accuracy
    total_accuracy += accuracy


    # Ask if the user wants to continue
    choice = input("Do you want to continue? (y/n): ")
    if choice.lower() != 'y':
        break
    
    # Update the total questions counter
    total_questions += 1

    # Calculate overall accuracy percentage
if total_questions > 0:
    overall_accuracy = total_accuracy / total_questions
    print(f"\nOverall accuracy of the model's evaluation: {overall_accuracy}")
else:
    print("No questions were answered.")


