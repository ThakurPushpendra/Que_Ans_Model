# Question-Answer Matching Model

This project implements a question-answer matching model using TF-IDF vectors and cosine similarity. It allows users to input their answers to questions and calculates the match percentage between their answers and pre-stored answers.

## Instructions

To run the code, you need to have Python 3 installed along with the following libraries:

- nltk
- scikit-learn (sklearn)

You can install the required libraries using pip:

**pip install nltk**

**pip install scikit-learn**


Additionally, make sure to download the NLTK stopwords corpus by running the following Python code:

```python
import nltk
nltk.download('stopwords')

The code also expects a text file containing the questions and answers in a specific format. You can specify the file path in the filename variable in the code. The file should have each question and answer pair on a separate line, separated by a tab character (/t). For example:

Question 1/tAnswer 1
Question 2/tAnswer 2
...

References
**TF-IDF Vectorization**
**Cosine Similarity**

