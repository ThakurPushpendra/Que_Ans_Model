import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocess text data
def preprocess_text(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Example question and answer data
questions = [
    "What is Python?",
    "What are the main features of Python?",
    "How does Python handle exceptions?",
    "What is an Interpreted language?"
]

answers = [
    "Python is a high-level programming language known for its simplicity.",
    "Python has a simple syntax, automatic memory management, and a large standard library.",
    "Python uses try-except blocks to handle exceptions.",
    "An Interpreted language executes its statements line by line. Languages such as Python, Javascript, R, PHP, and Ruby are prime examples of Interpreted languages."
]

# Preprocess question and answer data
preprocessed_questions = [preprocess_text(question) for question in questions]
preprocessed_answers = [preprocess_text(answer) for answer in answers]

# Build TF-IDF vectors for the preprocessed answers
vectorizer = TfidfVectorizer()
answer_vectors = vectorizer.fit_transform(preprocessed_answers)


# User input
for i, question in enumerate(questions):
    user_input = input(f"Enter your answer for question '{question}': ")

# Preprocess user input
preprocessed_user_input = preprocess_text(user_input)

# Convert user input to TF-IDF vector
user_vector = vectorizer.transform([preprocessed_user_input])

# Compute cosine similarity between user answer and stored answers
similarities = cosine_similarity(user_vector, answer_vectors)

# Find the highest similarity score
highest_similarity = max(similarities[0])
match_percentage = highest_similarity * 100

# Print the match percentage
print(f"Match percentage: {match_percentage:.2f}%")
print()
