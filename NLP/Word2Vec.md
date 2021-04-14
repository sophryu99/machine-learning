# Word2Vec

> Referred to <u>Stanford CS224N: NLP and Deep learning</u>

### Overview

- Word2Vec is a framework for learning word vectors



**Idea**:

- We have a **large corpus of text**
- Every word in a fixed vocabulary is represented by a vector
- Go through each position t in the text, which has a center word c and context words o
- Use the **similarity of the word vectors** for c and o to **calculate the probability** of o given c
- **Keep adjusting the word vectors** to maximize this probability



Example windows and process for computing

