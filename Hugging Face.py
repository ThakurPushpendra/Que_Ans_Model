from nltk.corpus import stopwords
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import random


# Preprocess text data
def preprocess_text(text, tokenizer):
    # Tokenize text
    tokens = tokenizer.tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Read questions and answers from a text file
def read_text_file(filename):
    questions = []
    answers = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                question, answer = line.split("/t")
                questions.append(question)
                answers.append(answer)

    return questions, answers



# User input
def process_user_input(questions, answers, model, tokenizer):
    match_percentages = []

    for i, question in enumerate(questions):
        user_input = input(f"Enter your answer for question '{question}': ")
        # Preprocess user input
        preprocessed_user_input = preprocess_text(user_input, tokenizer)

        # Tokenize and convert user input to input IDs
        inputs = tokenizer.encode_plus(
            preprocessed_user_input,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']

        # Generate BERT embeddings for user input
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

        # Convert embeddings to numpy array
        embeddings = embeddings.detach().numpy()

        # Compute cosine similarity between user answer and stored answers
        similarities = cosine_similarity([embeddings], answer_embeddings)

        # Find the highest similarity score
        highest_similarity = max(similarities[0])
        match_percentage = highest_similarity * 100

        match_percentages.append(match_percentage)
        

    #     # Print the match percentage
    # for i,question in enumerate(questions):
    #     print(f"Match percentage for question '{question}': {match_percentage:.2f}%")
    #     print()

    overall_match_percentage = sum(match_percentages) / len(questions)
    print(f"Overall Match percentage: {overall_match_percentage:.2f}%")

# Main code
if __name__ == '__main__':
    # Load BERT model and tokenizer
    print("------Bert Model runs from here")
    model_name = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Set model to evaluation mode
    model.eval()

    # Specify the file path of the question-answer pairs text file
    filename = '/home/tv-python-dev/Downloads/S09_question_answer_pairs.txt'

    # Read the questions and answers from the text file
    questions, answers = read_text_file(filename)

    # Preprocess and encode the answers using BERT tokenizer
    preprocessed_answers = [preprocess_text(answer, tokenizer) for answer in answers]
    answer_inputs = tokenizer.batch_encode_plus(
        preprocessed_answers,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    answer_input_ids = answer_inputs['input_ids']

    # Generate BERT embeddings for the answers
    with torch.no_grad():
        answer_outputs = model(answer_input_ids)
        answer_embeddings = answer_outputs.last_hidden_state.mean(dim=1)

    random.shuffle(questions)

    # Process user input
    process_user_input(questions, answers, model, tokenizer)
