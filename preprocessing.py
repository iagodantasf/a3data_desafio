import pandas as pd
import ast
from typing import Dict, Any


def preprocessed_books_rating(cfg: Dict[str, Any]) -> pd.DataFrame:
    books_rating = (
        pd.read_csv(cfg['path_read']['books_rating'])
        .rename(columns=cfg['col_renamer']['books_rating'])
        [cfg['useful_cols']['books_rating']]
        .dropna(subset=cfg['drop_nan']['books_rating'])
        .fillna(cfg['fill_nan']['books_rating'])
        .drop_duplicates()
    )
    if cfg['drop_eq_reviews_dif_time']:
        idx_to_drop = (
            books_rating
            .drop(columns='time')
            .duplicated()
            .pipe(lambda x: x[x]).index
        )
        books_rating = books_rating.drop(idx_to_drop)
    books_rating = books_rating.drop(columns='time')
    books_rating = (
        books_rating
        .groupby(['title', 'user_id']).agg({
            'score': 'mean',
            'summary': lambda x: '. '.join(x),
            'text': lambda x: '. '.join(x)
        })
        .reset_index()
    )
    books_rating['full_text'] = (
        books_rating['summary'] + '. ' + books_rating['text']
    )
    books_rating = books_rating.drop(columns=['summary', 'text'])
    full_text_len = books_rating['full_text'].str.len()
    max_full_text_len = (
        full_text_len
        .quantile(cfg['drop_text_above_quantile'])
    )
    return books_rating[full_text_len <= max_full_text_len]


def preprocessed_books_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    books_data = (
        pd.read_csv(cfg['path_read']['books_data'])
        .rename(columns=cfg['col_renamer']['books_data'])
        [cfg['useful_cols']['books_data']]
        .dropna(subset=cfg['drop_nan']['books_data'])
        .fillna(cfg['fill_nan']['books_data'])
        .drop_duplicates()
    )
    books_data['authors'] = (
        books_data['authors']
        .apply(
            lambda x:
            float('nan') if pd.isna(x) else
            ast.literal_eval(x)
        )
    )
    books_data['categories'] = (
        books_data['categories']
        .apply(
            lambda x:
            float('nan') if pd.isna(x) else
            ast.literal_eval(x)
        )
    )
    return books_data.explode('authors').explode('categories')
