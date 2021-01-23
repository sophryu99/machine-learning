# Natural Language Processing

[TOC]

> This repository contains basics of NLP theory and notes with code references to this link.

<br/>

## What is NLP?

NLP is a subfield of computer science and artificial intelligence concerned with interactions between computers and human languages. It is used to apply *machine learning* algorithms to *text* and *speech*. 

<br/>

NLP is used in:

- speech recognition
- document segmentation
- machine translation
- spam detection
- named entity recognition
- questions answering
- autocomplete, predictive typing

<br/>

Examples around us:

- Apple Siri, MS Cortana: recognizes natural voice commands
- Gmail: spam detection to filter out spam emails

<br/>

**NLTK Library for Python**

NLTK (Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. It provides interfaces to many **corpora** and **lexical resources**.

<br/>

## Basics of NLP for text



### 1. Sentence Tokenization

Dividing a string of written language into its **component sentences**.

<br/>

**Example**

> *Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice.*

<br/>

Use `nltk.sent_tokenize` function

```python
text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(sentence)
    print()
```

<br/>

Output:

```
Backgammon is one of the oldest known board games.

Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East.

It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice.
```

We can see that the paragraph is separated by "."

<br/>



### 2. Word Tokenization

Dividing a string of written language into its **component words**.

<br/>

Use `nltk.word_tokenize` function

```python
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print(words)
    print()
```

<br/>

Output:

```
['Backgammon', 'is', 'one', 'of', 'the', 'oldest', 'known', 'board', 'games', '.']

['Its', 'history', 'can', 'be', 'traced', 'back', 'nearly', '5,000', 'years', 'to', 'archeological', 'discoveries', 'in', 'the', 'Middle', 'East', '.']

['It', 'is', 'a', 'two', 'player', 'game', 'where', 'each', 'player', 'has', 'fifteen', 'checkers', 'which', 'move', 'between', 'twenty-four', 'points', 'according', 'to', 'the', 'roll', 'of', 'two', 'dice', '.']

```

We can see that the paragraph is separated by words. 

<br/>

### 3. Text Lemmatization and Stemming

For grammatical reasons, documents may can **different forms of a word** (eg. drives, drive, driving). Also, sometimes there are **related words** (eg. nation, national, nationality).

<br/>

Goal of both **stemming** and **lemmatization** is to **reduce inflectional forms** and derive a word to a **common base form**. 

<br/>

Examples:

- stemming: am, are, is => be
- lemmatization: Dog, dogs, dog's, dogs' => dog

<br/>

```python
def compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word, pos):
    """
    Print the results of stemmind and lemmitization using the passed stemmer, lemmatizer, word and pos (part of speech)
    """
    print("Stemmer:", stemmer.stem(word))
    print("Lemmatizer:", lemmatizer.lemmatize(word, pos))
    print()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "seen", pos = wordnet.VERB)
compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word = "drove", pos = wordnet.VERB)
```

<br/>

Output:

```
Stemmer: seen
Lemmatizer: see

Stemmer: drove
Lemmatizer: drive
```

<br/>

### 4. Stop words

Words which are **filtered** out **before** or **after** processing of text. During machine learning of a text, such words can add a lot of noise. Therefore, we aim to minimize such errors by removing irrelevant words.

<br/>

Example:

```python
from nltk.corpus import stopwords
print(stopwords.words("english"))
```

```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

<br/>

```python
stop_words = set(stopwords.words("english"))
sentence = "Backgammon is one of the oldest known board games."

words = nltk.word_tokenize(sentence)
without_stop_words = [word for word in words if not word in stop_words]
print(without_stop_words)

```

<br/>

Output (stop words removed):

```
['Backgammon', 'one', 'oldest', 'known', 'board', 'games', '.']
```

