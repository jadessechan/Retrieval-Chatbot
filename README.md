# Retrieval-Chatbot

This program uses tf-idf and cosine similarity to implement a rule-based chatbot. It's responses are based off of the [raft consensus algorithm paper](https://raft.github.io/raft.pdf)

*If the jupyter notebook is not rendering, visit [this link](https://nbviewer.jupyter.org/github/jadessechan/Retrieval-Chatbot/blob/master/main.ipynb)

## Getting started
1. clone or download this repository
```sh
https://github.com/jadessechan/Retrieval-Chatbot
```
2. open main.ipynb
3. run the code to chat with Bot about raft!

## Demo
You can enter key words or sentences related to Raft to get a clear enought response from Bot. The corpus- paper in our case- is large enough to not have the same sentence extracted as a reply from Bot.

Here is an example of a chat with Bot using both key words and direct questions:
![chat example](https://github.com/jadessechan/Retrieval-Chatbot/blob/master/imgs/chat_example.png)

## Implementation
I used Scikit-learn library's TfidfVectorizer and cosine_similarity to compare how similar the user input is to the raft corpus.
```sh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
First, I converted the paper into sentence tokens using NLTK's sent_tokenize to pass into the TfidfVectorizer:
```sh
article_sentences = nltk.sent_tokenize(article_text)
word_vectorizer = TfidfVectorizer(tokenizer=clean_input, stop_words='english')
```
The function fit_transform converts the tokens into vectors, that we can use to calculate cosine similarity with
```sh
all_word_vectors = word_vectorizer.fit_transform(article_sentences)
similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
```
After computing the cosine similarity of the paper vectors and user input, I sorted the list from least to greatest, with the highest cosine score at the end of the list. I grabbed the second to last vector because the last element is the user input.
```sh
similar_sentence_number = similar_vector_values.argsort()[0][-2]
```
And finally, I passed the index of the most similar sentence in the "article_sentences" list and appended that to the "bot_response", thus creating the reply to the user.
```sh
bot_response = ''
bot_response += article_sentences[similar_sentence_number]
```
