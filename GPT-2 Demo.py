from transformers import pipeline

# Load the pre-trained model for analysis
analysis_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load the pre-trained model for question generation
question_model = pipeline("text-generation", model="gpt2")

# Define the interview questions
interview_questions = [
    "Tell me about yourself.",
    "What are your strengths?",
    "What are your weaknesses?",
    "Why do you want to work here?",
    "Can you describe a challenging situation you faced at work and how you handled it?",
    "Where do you see yourself in five years?"
]

# Define a function to generate AI analysis/assessment
def generate_analysis(answer):
    analysis = analysis_model(answer)[0]
    return analysis["label"], analysis["score"]

# Define a function to generate follow-up questions
def generate_follow_up_question(answer):
    question = question_model(answer, max_length=20)[0]["generated_text"]
    return question

# Main interview loop
def conduct_interview():
    print("Welcome to the AI Interviewer!")
    print("Please answer the following questions:\n")
   
    for i, question in enumerate(interview_questions):
        print(f"Question {i+1}: {question}")
        answer = input("Your answer: ")
       
        # Generate AI analysis
        label, score = generate_analysis(answer)
       
        # Print AI analysis
        print(f"\nAI Analysis: {label} (Confidence: {score:.2f})")
       
        # Generate follow-up question
        if i < len(interview_questions) - 1:
            follow_up_question = generate_follow_up_question(answer)
            print(f"Follow-up Question: {follow_up_question}\n")

# Start the interview
conduct_interview()