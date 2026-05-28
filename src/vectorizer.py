'''
Build vocabulary (manual BoW)
Convert text → vectors
'''

# ----------- BoW (Manual) -----------
#Building a simple BoW vocab
def build_vocab(tokenized_texts, max_features=5000):
    # Count word frequencies
    word_freq={}

    for tokens in tokenized_texts:
        for word in tokens:
            if word in word_freq:
                word_freq[word]+=1
            else:
                word_freq[word]=1
    
    #sort by frequency (descending)
    sorted_words=sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    #keep only top words
    vocab=[word for word, freq in sorted_words[:max_features]]

    return vocab

def create_vocab_index(vocab):
    return {word: idx for idx, word in enumerate(vocab)}
    
#convert TEXT → VECTOR
def text_to_vector(tokens, vocab_index):
    vector= [0] * len(vocab_index)

    for word in tokens:
        if word in vocab_index:
            index= vocab_index[word] #find position
            vector[index] +=1 #increment count
    return vector

#convert list of tokenized texts to matrix
def texts_to_matrix(tokenized_texts, vocab):
    vocab_index = create_vocab_index(vocab)
    matrix=[]

    for tokens in tokenized_texts:
        vector=text_to_vector(tokens, vocab_index)
        matrix.append(vector)

    return matrix

#main function to get vectorized dataset
def vectorized_dataset(tokenized_texts, vocab):
    return texts_to_matrix(tokenized_texts, vocab)


# ----------- TF-IDF -----------

# TF-IDF vectorization (using sklearn)
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorized_dataset(
    train_texts,
    test_texts,
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer