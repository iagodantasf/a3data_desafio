import nltk
import numpy as np
import multiprocessing as mp
import textstat
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List


nltk.download("punkt")


def readability(
    reviews: List[str],
) -> List[float]:
    """
    Return a list of readability scores for each review in the input list.
    """
    return [
        np.clip(textstat.flesch_reading_ease(review)/100, 0, 1)
        for review in reviews
    ]


def relevance_bert(
    reviews: List[str],
    book_description: str,
) -> List[float]:
    """
    Return a list of relevance scores for each review in the input list.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    book_embedding = model.encode(book_description)
    review_embeddings = model.encode(reviews)
    return [
        util.cos_sim(book_embedding, review_embedding).item()
        for review_embedding in review_embeddings
    ]


def relevance_word2vec(
    reviews: List[str],
    language: str = "english",
    vector_size: int = 100,
    workers: int = mp.cpu_count(),
) -> List[float]:
    """
    Return a list of relevance scores for each review in the input list.
    """
    stop_words = set(stopwords.words(language))
    tokenized_reviews = [
        word_tokenize(review.lower()) for review in reviews
    ]
    filtered_tokens = [
        [word for word in tokens if word not in stop_words]
        for tokens in tokenized_reviews
    ]
    model = Word2Vec(
        sentences=filtered_tokens,
        vector_size=vector_size,
        workers=workers,
    )

    def get_review_vector(review):
        word_vectors = [
            model.wv[word]
            for word in review
            if word in model.wv
        ]
        return (
            np.mean(word_vectors, axis=0) if word_vectors
            else np.zeros(vector_size)
        )

    review_vectors = np.array([
        get_review_vector(review)
        for review in filtered_tokens
    ])
    return np.linalg.norm(review_vectors, axis=1)
