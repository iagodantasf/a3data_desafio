from textblob import TextBlob
from typing import Tuple


def sentiment(text: str, model: str = 'textblob') -> Tuple[float, float]:
    """
    Return a tuple of form (polarity, subjectivity) where polarity is a
    float within the range [-1.0, 1.0] and subjectivity is a float
    within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is
    very subjective.
    """
    blob = TextBlob(text)
    return blob.sentiment
