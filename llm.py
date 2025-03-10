import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from typing import List, Literal


def summarize_review_t5(
    title: str,
    description: str,
    authors: str,
    categories: str,
    score: str,
    review: str,
    model_name: Literal['t5-small', 't5-base', 't5-large'],
) -> str:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    prompt = (
        f"The information below is about a book and a review of it:"
        f"\n\nTitle of the book: {title}"
        f"\n\nDescription of the book: {description}"
        f"\n\nAuthors of the book: {authors}"
        f"\n\nCategories of the book: {categories}"
        f"\n\nScore given in the review: {score} / 5.0"
        f"\n\nThe review of the book: {review}"
        f"\n\nSummarize the review."
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=10000, truncation=True
    )
    output = model.generate(**inputs, max_length=500)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def summarize_reviews_mistral(
    reviews: List[str],
    by: Literal['title', 'author', 'category'],
) -> str:
    qa_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=torch.device("cpu"),
    )
    context = " ".join(reviews)
    if by == 'title':
        question = (
            'The reviews below are from the same book.'
            ' What do the reviewers think about this book?'
        )
    elif by == 'author':
        question = (
            'The reviews below are from books of the same author.'
            ' What do the reviewers think about this author?'
        )
    elif by == 'category':
        question = (
            'The reviews below are from books of the same category.'
            ' What do the reviewers think about this category?'
        )
    else:
        raise ValueError(f"Invalid value for 'by': {by}")
    prompt = f"{question}\n\nReviews: {context}"
    response = qa_pipeline(prompt, max_length=200, do_sample=True)
    return response[0]["generated_text"]
