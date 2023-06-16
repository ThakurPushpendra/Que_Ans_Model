
import spacy

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

class Question:
    def __init__(self, text, answer):
        self.text = text
        self.answer = answer

    def check_answer(self, user_answer):
        # Calculate the similarity score between the user's answer and the correct answer
        similarity_score = nlp(self.answer).similarity(nlp(user_answer))
       
        # Set a similarity threshold (adjust as needed)
        similarity_threshold = 0.8
       
        # Return True if the similarity score exceeds the threshold, indicating a correct answer
        return similarity_score >= similarity_threshold


class Quiz:
    def __init__(self):
        self.questions = []
        self.score = 0

    def add_question(self, question):
        self.questions.append(question)

    def run_quiz(self):
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


# Create the quiz questions
question1 = Question("What is React.js?", """react.js is an open-source JavaScript library that simplifies the process of building interactive user interfaces. It provides a component-based approach, allowing developers to create reusable UI elements and efficiently manage state and data flow.
React.js is a JavaScript library designed to make it easier to build user interfaces. It focuses on component reusability and provides a virtual DOM to optimize rendering performance. By using React, developers can create complex web applications with ease.
React.js is a powerful tool for creating dynamic and responsive web applications. It enables developers to break down the user interface into reusable components, which can be composed together to build complex UIs. React's efficient rendering engine ensures that only the necessary components are updated when data changes.
React.js is a JavaScript library developed by Facebook for building fast and scalable web applications. It introduces a declarative approach, where developers describe how the UI should look based on the current state, and React takes care of updating the actual UI efficiently.

.""")

# question2 = Question("Explain the concept of supply and demand in economics.", "Supply and demand is a fundamental concept in economics that explains the relationship between the availability of a product or service (supply) and the desire or need for that product or service (demand). In a competitive market, the price and quantity of a product are determined by the interaction of supply and demand. When supply is high and demand is low, prices tend to decrease. Conversely, when demand is high and supply is low, prices tend to increase. The equilibrium point where supply and demand intersect determines the market price and quantity.")

# Create the quiz
quiz = Quiz()
quiz.add_question(question1)
# quiz.add_question(question2)

# Run the quiz
quiz.run_quiz()