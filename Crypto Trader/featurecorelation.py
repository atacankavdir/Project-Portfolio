#/usr/bin/env python
#coding: utf-8

import pandas as pd
import numpy as np
np.random.seed (0)
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.base import Basestimator, TransformerMixin
from collections import Counter


class Corelation_eliminate(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7, corr_type='pearson'):
        self.threshold = threshold
        self.corr_type = corr_type
    
    def fit(self, X, y=None):
        """ Fit data to create a correlation matrix and eliminate highly correlated pair member with the higher correlation average.

        Args:
            self: self
            X (pandas.DataFrame): train dataset
                    Y (pandas.Series) : train target data
        
        Returns:
            self: self
        """
        #self.length = len(X)
        self.data = pd.DataFrame(X)
        
        if len(self.data)>50000:
            self.data = self.data.sample(n=50000, replace=False, random_state=1)
        
        self.corr_matrix = self.data.corr(method=self.corr_type).abs()
        
        self.upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(np.bool) )
        
        self.feat_1 = []
        self.feat_2 = []
        self.corr_list = []
        for i in range (len(self.upper.columns)):
            for j in range(i+l, len(self.upper.columns)):
                if self.upper.iloc[i,j] >= self.threshold:
                    self.feat_1.append(self.upper.index[i])
                    self.feat_2.append(self.upper.columns[j]
                    self.corr_list.append(self.upper.iloc[i,j])

        self.corr_pairs = pd.DataFrame({'featl' : self.feat_1,'feat2': self.feat_2,'corr' : self.corr_1ist})
        self.cor_mean = self.corr_matrix.mean()
        
        #corr removal
        self.feature_list = self.corr_matrix.columns
        self.removed_feat = []
        for i in range(0,len(self.corr_pairs)):
            if (self.corr_pairs.iloc[i,0] in self.feature_list) and (self.corr_pairs.iloc [i,1] in self.feature_list):
                if self.cor_mean.loc[self.corr_pairs.iloc[i,0]] < self.cor_mean.loc[self.corr_pairs.iloc[i,1]]:
                    self.removed_feat.append(self.corr_pairs.iloc[i,1])
                    self.feature_list = self.feature_list.drop(self.corr_pairs.iloc[i,1])
                else:
                    self.removed_feat.append(self.corr_pairs.iloc[i,0])
                    self.feature_list = self.feature_list.drop(self.corr_pairs.iloc[i,0])
        
        self.feat_list = self.corr_matrix.columns.isin(self.feature_list)
        
        return self
        
    def transform(self, X, y=None):
        """ Transform data according to chosen features in fit function.
        Args:
            self : self
            X(pandas.DataFrame) : train dataset
            Y(pandas.Series) :train target data
        Returns:
            X: pandas.DataFrame with selectted feature columns
        """
        return X[:,self.feat_list]
        
    def get_support(self):
        """Get feature list from fit function as a list.
        
        Args:
            self : self
        Returns:
            self.feat_list: list for selected features
        """
        return self.feat_list