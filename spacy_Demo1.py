import spacy
import pandas as pd
import random

# Load the spaCy English language model
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_lg")

class Question:
    def __init__(self, text, answers):
        self.text = text
        self.answers = answers

    def check_answer(self, user_answer):
        # Calculate the similarity score between the user's answer and each possible answer
        similarity_scores = [nlp(answer).similarity(nlp(user_answer)) for answer in self.answers]

        # Set a similarity threshold (adjust as needed)
        similarity_threshold = 0.8

        # Return True if any of the similarity scores exceed the threshold, indicating a correct answer
        return any(score >= similarity_threshold for score in similarity_scores)


class Quiz:
    def __init__(self):
        self.questions = []
        self.score = 0

    def add_question(self, question):
        self.questions.append(question)

    def load_questions_from_excel(self, file_path):
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Iterate over the rows and create Question objects
        for _, row in df.iterrows():
            question_text = row['Question']
            answers = [row['Answer1'], row['Answer2'], row['Answer3'], row['Answer4']]
            question = Question(question_text, answers)
            self.add_question(question)

    def run_quiz(self):
        random.shuffle(self.questions) 
        for question in self.questions:
            user_answer = input(question.text + " ")
            if question.check_answer(user_answer):
                self.score += 1

        self.display_score()

    def display_score(self):
        total_questions = len(self.questions)
        percentage = (self.score / total_questions) * 100
        print(f"You scored {self.score} out of {total_questions}.")
        print(f"Your percentage is: {percentage}%.")


# Create the quiz
quiz = Quiz()

# Load questions from an Excel file
quiz.load_questions_from_excel("/home/tv-python-dev/Documents/DemoQA.xlsx")

# Run the quiz
quiz.run_quiz()
