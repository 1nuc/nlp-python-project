import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import polars.selectors as cs
from scipy import stats
import numpy as np
from wordcloud import WordCloud

class TextClassification:
    def __init__(self,data):
        self.data=data
        plt.style.use('ggplot')

    def barplot_seaborn(self,x,y):
        plot=sns.barplot(
            data=self.data, 
            x=x, 
            y=y, 
            hue=y, estimator='sum')
        
    def WorldCloud(self):
        text=' '.join(self.data.select('OriginalTweet').to_series().to_list())
        wordcount=WordCloud(height=400, width=800,background_color='white')
        words=wordcount.generate(text)
        image=words.to_image()
        image_array=np.array(image)
        plt.figure(figsize=(20,8))
        plt.imshow(image_array, interpolation="bilinear")
        plt.axis('off')
        plt.show()
# used
    def pie(self, x, y):
        plt.figure(figsize=(20,8))
        plt.pie(self.data[x], labels=self.data[y], autopct='%1.1f%%', shadow=True, startangle=90)
        plt.show()

  