import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
import pickle as pkl
import csv
import re
# dot_token = r'\W{2,}'
# http_token = r'[a-zA-z]+://[^\s]*'
# # token_pattern = re.compile(r'(?u)(?<![#@])\b\w\S+\b')
# token_pattern = re.compile(r'(?u)[@|#]\b\w\S+\b')
# repeat_token = re.compile(r"(.)\1{2,}")


def tweet_process(text, tokenizer):
    processed_text = []
    for text_ in text:
        text_ = re.sub(r'@[\w]*', '', text_)  # Removing Twitter Handles (@user)
        text_ = re.sub(r'http\S+', '', text_)  # Removing URLs from text
        text_ = re.sub(r'[^A-Za-z#]', ' ', text_)  # Removing Punctuations, Numbers, and Special Characters
        text_ = tokenizer.tokenize(text_.lower())
        if len(text_) > 0:
            processed_text.append(text_)
        # print(processed_text)
    return processed_text


def text_preprocess(df_user, user_type):
    if user_type == 'train':
        start = 0
    elif user_type == 'valid':
        start = 5685
    elif user_type == 'test':
        start = 5685 + 1895
    word_dict = {}
    tnkzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    for idx in tqdm(range(len(df_user))):
        user = df_user['user'][idx]
        tweets = df_user['text'][idx]
        tweet_list = tweets.split('|||')

        processed_tweets = tweet_process(tweet_list, tnkzr)
        word_dict[idx+start] = processed_tweets

    return word_dict


if __name__ == '__main__':
    user_types = ['train', 'valid', 'test']
    for user_type in user_types:
        df_user = pd.read_csv('../dataset/cmu/user_info.{}.csv'.format(user_type),
                              names=['user', 'lat', 'lon', 'text'], header=None, sep='\t')
        word_dict = text_preprocess(df_user, user_type)

        with open(f'../dataset/cmu/word_emb_text.{user_type}.pkl', 'wb') as f:
             pkl.dump(word_dict, f)