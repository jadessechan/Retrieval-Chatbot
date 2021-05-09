# Retrieval-Chatbot

This program uses tf-idf and cosine similarity to implement a rule-based chatbot. It's responses are based off of a csv sheet I made with FAQs and answers to each.

You can look at either main.ipynb or main.py for the source code, depending on your preference.

## Getting started
1. clone or download this repository
```sh
https://github.com/jadessechan/Retrieval-Chatbot
```
2. open main.ipynb or main.py
3. run the code to chat with Bot to learn more about Rhodes' CS tutoring!

## Demo
The data set is composed of questions grouped into 3 that will have the same answer. If the user input aligns closely with the questions in the dataset, then
Bot will output the corresponding answer!

Here is an example of a chat with Bot where I asked questions identical to the one in the dataset and others with key words (you can also view this on [main.ipynb](https://github.com/jadessechan/Retrieval-Chatbot/blob/master/main.ipynb):

**When does tutoring start?**<br />
>Tutoring begins 5-11pm CDT from Sunday to Thursday (excluding school holidays) using the queue app.<br />

**Are there any rules for tutoring?**<br />
>Tutors are asked to limit time for each individual tutoring session to 10-20 minutes, since we have over 150+ students in 141/142/241 and 9 tutors. Tutoring is restricted to tutoring hours only, and only available using the queue app. Please do NOT DM tutors directly.<br />

**Are there expectations for tutoring?**<br />
>Tutoring is first-come first-serve for 141, 142, 241 only. Tutors will help you work through concepts, debug, and provide resources for further information.<br />

As expected, the first two questions output the correct response from Bot because the vectors and thus cosine similarity are identical to the ones in the data set. However, the last question doesn't output the associated response. I mentioned a key word, 'expectation', but Bot's answer was associated to the questions "What is tutoring?" and "Tell me about tutoring?" in the data.

## Implementation
I used Scikit-learn library's TfidfVectorizer and cosine_similarity to compare how similar the user input is to the *Questions* column in my data.
```sh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
First, I converted the *Questions* column into a long string, then converted it into sentence tokens using NLTK.
```sh
data = pd.read_csv('tutoring_data.csv')
q_string = ''
for i, row in data.iterrows():
    q_string = q_string + row.loc['Question'] + " "
    
q_tokens = nltk.sent_tokenize(q_string)
```
Then I initialized a TfidfVectorizer and called fit_transform to convert the tokens into vectors, which we can use to calculate the cosine similarity.
```sh
word_vectorizer = TfidfVectorizer(tokenizer=clean_input, stop_words='english')
all_word_vectors = word_vectorizer.fit_transform(q_tokens)
similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
```
After computing the cosine similarity of the user input and question vectors, I sorted the list from least to greatest, with the highest cosine score at the end of the list. ```sh[0][-2]``` indicates the most similar question to the user input, because I appended the user input to q_tokens, so the rather than grabbing the last element I got the second to last.
```sh
similar_sentence_number = similar_vector_values.argsort()[0][-2]
```
And finally, I passed the index of the most similar question to get the appropriate response in the *Answer* column in my data.
```sh
bot_response = ''
bot_response += data.at[similar_sentence_number, 'Answer']
```
