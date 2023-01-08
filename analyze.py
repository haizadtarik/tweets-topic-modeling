from dotenv import load_dotenv
import requests
import os
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import gensim
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

def list_params():
    return (
    """
    Tweet parameters. Available options are:
    attachments, author_id, context_annotations, conversation_id, created_at, entities, 
    geo, id, in_reply_to_user_id, lang, non_public_metrics, organic_metrics, possibly_sensitive, 
    promoted_metrics, public_metrics, referenced_tweets, source, text, and withheld
    """
    )
    
def create_url(tweet_fields,tweet_id):
    quote_tweet_url = f"https://api.twitter.com/2/tweets/search/recent?tweet.fields={tweet_fields}&query=url:{tweet_id}&max_results=100"
    replies_url = f"https://api.twitter.com/2/tweets/search/recent?tweet.fields={tweet_fields}&query=conversation_id:{tweet_id}&max_results=100"
    return quote_tweet_url, replies_url

def connect_to_endpoint(url):
    bearer_token = os.environ.get("BEARER_TOKEN")
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def retrive_data(id, params):
    load_dotenv()
    quote_tweet_url, replies_url = create_url(params,id)
    replies_json_response = connect_to_endpoint(replies_url)
    quoted_json_response = connect_to_endpoint(quote_tweet_url)
    replies_df = pd.DataFrame.from_dict(replies_json_response['data'])
    quoted_df = pd.DataFrame.from_dict(quoted_json_response['data'])
    df = pd.concat([replies_df,quoted_df])
    return df

def clean_stopwords(text):
    stop_words = stopwords.words('english')
    return " ".join([word for word in str(text).split() if word not in stop_words])

def clean_punctuations(text):
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def clean_hyperlinks(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+)|(@[^\s]+))',' ',data)

def clean_numbers(data):
    return re.sub('[0-9]+', '', data)

def stemming_on_text(data):
    st = nltk.PorterStemmer()
    text = [st.stem(word) for word in data]
    return data

def lemmatizer_on_text(data):
    lm = nltk.WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in data]
    return data

def preprocess_data(tweet_df):
    tokenizer = RegexpTokenizer(r'\w+')
    tweet_df['text'] = tweet_df['text'].str.lower()
    tweet_df['text'] = tweet_df['text'].apply(lambda text: clean_hyperlinks(text))
    tweet_df['text'] = tweet_df['text'].apply(lambda text: clean_stopwords(text))
    tweet_df['text']= tweet_df['text'].apply(lambda text: clean_punctuations(text))
    tweet_df['text'] = tweet_df['text'].apply(lambda text: clean_numbers(text))
    tweet_df['text'] = tweet_df['text'].apply(tokenizer.tokenize)
    tweet_df['text']= tweet_df['text'].apply(lambda text: stemming_on_text(text))
    tweet_df['text'] = tweet_df['text'].apply(lambda text: lemmatizer_on_text(text))
    return tweet_df['text'] 


def lda_modeling(texts, num_topics=5):
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary)
    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'topics.html')


import argparse

def args_parse():
    parser = argparse.ArgumentParser(description='Perform topic modeling on tweets using LDA')
    parser.add_argument('--id', type=int, help='Tweet ID')
    parser.add_argument('--params', nargs='+', help=list_params())

    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    id = args.id
    params = ','.join(args.params)
    tweet_df = retrive_data(id, params)
    processed_tweets = preprocess_data(tweet_df)
    lda_modeling(processed_tweets)
    

