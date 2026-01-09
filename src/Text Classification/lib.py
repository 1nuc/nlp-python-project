import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import polars.selectors as cs
from scipy import stats


class TextClassification:
    def __init__(self,data):
        self.data=data
        plt.style.use('ggplot')
        
    def boxplot_exp(self):
        fig, ax=plt.subplots(self.dev_count, 1, figsize=(10, 7 * self.dev_count))
        for i,x in enumerate(self.devices):
            plot=sns.boxplot(data=self.data, x=x, ax=ax[i])
            xlabel=x.removeprefix("out.electricity.")
            plot.set(xlabel=xlabel, ylabel="count")
    
    def hist_exp(self):
        fig, ax=plt.subplots(self.dev_count, 1, figsize=(10, 7 * self.dev_count))
        for i,x in enumerate(self.devices):
            plot=sns.histplot(data=self.data, x=x, ax=ax[i])
            xlabel=x.removeprefix("out.electricity.")
            plot.set(xlabel=xlabel, ylabel="count")
                
    def boxen_exp(self):
        fig, ax=plt.subplots(self.dev_count, 1, figsize=(10, 7 * self.dev_count))
        for i,x in enumerate(self.devices):
            plot=sns.boxenplot(data=self.data, x=x, ax=ax[i])
            xlabel=x.removeprefix("out.electricity.")
            plot.set(xlabel=xlabel, ylabel="count")
    
    def edit_column_names(self,df):
        df=df.rename(lambda col: col.replace('in.','') if col.startswith('in.') else col.replace('.energy_consumption..kwh', ''))
        return df

    def barplot_seaborn(self,x,y):
        plot=sns.barplot(
            data=self.data, 
            x=x, 
            y=y, 
            hue=y, estimator='sum')
        
    def test_corr(self,df,var):
        cols=[col for col in df.columns if col.startswith('out.electricity') and col != var]
        arr=[]
        for col in cols:
            p_corr, p_val=stats.pearsonr(df[col], df[var])
            df_test=pd.DataFrame({
                "p_value":[p_val],
                "pearson correlation:": [p_corr],
                "col": [col]
            })
            arr.append(df_test)
        return pd.concat(arr)
    
    def display_totals(self,df, x, y):
        plt.figure(figsize=(20,8))
        sns.jointplot(data=df, x=x, y=y, kind='reg')
        
    # visualizing long duration devices per each hour of the day by having a barplot
    def long_dev_temporal_based(self,df, temporal_type):
        cols=[col for col in df.columns if col.startswith('out.electricity')]
        for col in cols:
            plt.figure(figsize=(20,8))
            plot=sns.barplot(data=df, x=temporal_type, y=col, color='purple')
            label=col.removeprefix('out.electricity.')
            plot.set(xlabel=temporal_type, ylabel=label)
            plt.show()
            
     
    def _barplot_seaborn(self,data,x,y, hue_var=None):
        plt.figure(figsize=(20, 8))
        if hue_var !=None:
            plot=sns.barplot(
                data=data, 
                x=x, 
                y=y, 
                hue=hue_var, estimator='sum')
        else :
            plot=sns.barplot(
                data=data, 
                x=x, 
                y=y, 
                hue=y, estimator='sum')
        plt.show()
    
    def lineplot(self, data, x,y):
        plt.figure(figsize=(20,8))
        if isinstance(y, list):
            for device in y:
                ylabel=device.removeprefix('out.electricity.')
                if device.startswith('out.site_energy.'):
                    ylabel=device.removeprefix('out.')
                plot=sns.lineplot(data=data, x=x ,y=device, label=ylabel)
        else:
            ylabel=y.removeprefix('.out.electricity')
            sns.lineplot(data=data, x=x ,y=y, label=ylabel, color='b')
        plt.show()

    def replot_zone(self, data, x,y,labels,hue_var):
        plt.figure(figsize=(20,8))
        sns.relplot(
            data=data, x=x, y=y,
            col=hue_var, hue=labels, style=labels,height=4,
            kind="line", col_wrap=2 
        )
        plt.show()
    
    def pairplot(self,data, hue_var=None):
        plt.figure(figsize=(20, 8))
        sns.pairplot(data=data, hue=hue_var)
        plt.show()

    def heatmap(self,data, labels):
        plt.figure(figsize=(20, 8))
        plot=sns.heatmap(data, annot=True, fmt=".3f", linewidth=.5, xticklabels=True, yticklabels=True)
        plot.set_xticklabels(labels)
        plot.set_yticklabels(labels, rotation=0)
        plt.show()
# used
    def pie(self, x, y):
        plt.figure(figsize=(20,8))
        plt.pie(self.data[x], labels=self.data[y], autopct='%1.1f%%', shadow=True, startangle=90)
        plt.show()

    def catplot(self, data, x,y, hue_var, col_var):
        plt.figure(figsize=(20,8))
        sns.catplot(data=data, x=x, y=y, hue=hue_var, col=col_var,
                    col_wrap= 2, kind='bar',  aspect=2, height=5)
        plt.show()

    def test_corr_anova(self, df, grp_col, target_col):
        levels=df.select(pl.col(grp_col).unique()).to_series().to_list()
        arr=[]
        for lev in levels:
            grp=df.filter(pl.col(grp_col)==lev).select(pl.col(target_col)).to_series().to_list()
            arr.append(grp)
        anova_test=stats.f_oneway(*arr)
        d=pl.DataFrame({
                "Anova test statistic": [anova_test.statistic],
                "p-value":[anova_test.pvalue]
        })
        return d