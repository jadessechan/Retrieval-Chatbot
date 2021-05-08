import nltk
import string
import re
import random
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    data = pd.read_csv('tutoring_data.csv')
    q_string = ''
    for i, row in data.iterrows():
        q_string = q_string + row.loc['Question'] + " "

    q_string = q_string.lower()
    # remove special characters and empty spaces
    q_string = re.sub(r'\[[0-9]*\]', ' ', q_string)
    q_string = re.sub(r'\s+', ' ', q_string)

    sent_tokens = nltk.sent_tokenize(q_string)

    continue_dialogue = True
    print("Hello, I am bot.")
    while (continue_dialogue == True):
        human_text = input()
        human_text = human_text.lower()
        if human_text != 'bye':
            if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
                continue_dialogue = False
                print("bot: Most welcome")
            else:
                if generate_greeting_response(human_text) != None:
                    print("bot: " + generate_greeting_response(human_text))
                else:
                    print("bot: ", end="")
                    print(generate_response(human_text, data, sent_tokens))
                    sent_tokens.remove(human_text)
        else:
            continue_dialogue = False
            print("bot: Good bye and take care of yourself...")


def generate_greeting_response(greeting):
    greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
    greeting_responses = ["hey", "hi", "*nods*", "hello", "Welcome!"]

    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)


# pre-process user input along with text
def clean_input(user_input):
    # remove punctuation
    user_input = user_input.translate(str.maketrans(' ', ' ', string.punctuation))
    # tokenize input
    tokens = nltk.word_tokenize(user_input)
    wnl = nltk.stem.WordNetLemmatizer()

    for words in tokens:
        # lemmatize words
        wnl.lemmatize(words)
    return tokens


# take user input, find cosing similarity of input, and compare with sentence tokens
def generate_response(user_input, data, sent_tokens):
    bot_response = ''
    sent_tokens.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=clean_input, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(sent_tokens)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "I am sorry, I could not understand you"
        return bot_response
    else:
        # output corresponding answer from answer column
        bot_response += data.at[similar_sentence_number, 'Answer']
        return bot_response


main()

