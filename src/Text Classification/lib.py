import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os
import nltk
from collections import Counter
import polars.selectors as cs
from wordcloud import WordCloud
import spacy_cleaner
from spacy_cleaner import Cleaner
from spacy_cleaner.processing import removers, mutators
import string
import spacy
from nltk.stem import PorterStemmer

class TextClassification:
    def __init__(self,data):
        self.data=data
        plt.style.use('ggplot')
        self.nlp=spacy.load('en_core_web_sm')
        self.spacy_pipeline=spacy_cleaner.Cleaner(
            self.nlp,
            removers.remove_email_token,
            removers.remove_stopword_token,
            removers.remove_url_token,
            mutators.mutate_lemma_token,
            removers.remove_punctuation_token)

        self.stop_words=nltk.corpus.stopwords.words('english')

    ## Data Preprocessing ----------------------------------------------- 
    def preprocess_spacy(self,text): # Preprocess the data using spacy
        # spliiting by @, then rejoining again and form strings
        text=[''.join(text.split('@'))]
        # define the pipeline for cleaning text
        return(self.pipeline.clean(text))
    
    def preprocess_pl_native(self,v): # Preprocess the data using polars native expression
        # S non whiteshapce character --s whitespace character 
        data = v.with_columns(
                pl.col('OriginalTweet').str.to_lowercase().str.replace_all(r'http\S+|www\S+|@|#', '').str.replace_all(r'[^\w\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars())
        return data
    
    # removal of stop words and punctuations
    def remove_stop_words_and_punc(self,t):
        punc=string.punctuation
        list_refined=[word.lower() for word in t if word.lower() not in self.stop_words and word not in punc]
        return ' '.join(list_refined)
    
    def porter_stem(self,t) -> list:
        stem_porter=PorterStemmer()
        return ''.join([stem_porter.stem(word) for word in t])

   
    # Data Exploration ------------------------------------------------------------------------
    def barplot_seaborn(self,x,y):
        plt.figure(figsize=(20,8))
        plot=sns.barplot(
            data=self.data, 
            x=x, 
            y=y, 
            hue=y, estimator='sum')
    
    def boxplot(self):
        words_length=self.data.with_columns(
            pl.col('OriginalTweet').str.split(' ').list.len().alias("Text Length"))
        plt.figure(figsize=(20,8))
        sns.boxplot(data=words_length, x="Sentiment", y="Text Length")
        plt.title(f"Text length distribution for each sentiment")
        plt.show()
        
    def WorldCloud(self, factor=None):
        data=self.data
        if factor is not None:
            data.filter(pl.col('Sentiment')==factor)
        text=' '.join(data.select('OriginalTweet').to_series().to_list())
        wordcount=WordCloud(height=400, width=800,background_color='white')
        words=wordcount.generate(text)
        image=words.to_image()
        image_array=np.array(image)
        plt.figure(figsize=(20,8))
        plt.imshow(image_array, interpolation="bilinear")
        plt.axis('off')
        plt.show()
        
    def Top_words(self, factor):
        data=self.data.filter(pl.col("Sentiment")==factor)
        words=''.join(data.select(pl.col('OriginalTweet')).to_series().to_list())
        words_counter=Counter(words.split())
        most_common=words_counter.most_common(20)
        words, counter=zip(*most_common)
        plt.figure(figsize=(20,8))
        plt.barh(words, counter)
        plt.title(f"Top words for {factor} class")
        plt.show()

    def pie(self, x, y):
        plt.figure(figsize=(20,8))
        plt.pie(self.data[x], labels=self.data[y], autopct='%1.1f%%', shadow=True, startangle=90)
        plt.show()

  