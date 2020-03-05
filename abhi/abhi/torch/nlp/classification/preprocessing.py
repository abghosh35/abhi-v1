import numpy as np
from bs4 import BeautifulSoup as BS
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as cntvect
from sklearn.preprocessing import LabelBinarizer as LBL


def Label2Index(labels, onehot_encoded=True):
    lbin = LBL()
    lbin.fit(labels)

    _labels_ = lbin.transform(labels)

    return _labels_

class BasicPreProcessing:
    """
    This module performs pre-processing of the raw texts and labels for NER exercise. Note that this works at WORD level.
    Inputs: 
        texts: List of raw texts of sentences
        for_embedding: Boolean (default=True) Return data formated 
        ngrams: tuple: (Default=(1, 1)): A tuple of minimum and maximum value of ngrams provided as:(min_ngram_range, max_ngram_range): 
        min_df: float (< 1): (default=0.005): Minimum percentage of occurrence of a word to be considered
        max_df: float (< 1): (default=0.99): Maximum percentage of occurrence of a word to be considered
        max_features: (default: all): Maximum number of features to be considered from the vocabulary list. 
                                      Possible values are: 'all' or integer
        
    Returns: List of cleaned and Pre-processed string
    """

    def __init__(self, texts, for_embedding=True, ngrams=(1, 1), min_df=0.005, max_df=0.99, max_features='all', padding_text='<PAD>'):
        self.texts = pd.DataFrame({'RawTexts': texts})
        self.for_embedding = for_embedding
        self.ngrams = ngrams
        self.punctuations = string.punctuation
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.padding_text = padding_text

    def FetchData(self):
        if self.for_embedding:
            return self._ReturnDataForEmbedding()
        else:
            return self._CleanText()
    
    def _CleanText(self):
        self.texts = self.texts['RawTexts'].apply(lambda x: self._LowerText(self._RemovePunctuations(BS(x, 'html.parser').get_text())))
        
        return self.texts.tolist()
        
    def _RemovePunctuations(self, text):
        for p in self.punctuations:
            text = text.replace(p, ' ').replace('  ', ' ').strip()
            
        return text
    
    def _LowerText(self, text):
        return text.lower()

    
    def _ReturnDataForEmbedding(self):
        texts = self._CleanText()
        if self.max_features == 'all':
            self.max_features = np.max([len(x) for x in texts])

        CountVect = cntvect(min_df=self.min_df, max_df=self.max_df, max_features=self.max_features)
        CountVect.fit(texts)
        vocabulary = CountVect.vocabulary_

        word2idx = {w: i for i, w in enumerate(vocabulary)}
        idx2word = {i: w for i, w in enumerate(vocabulary)}

        word2idx[self.padding_text] = len(vocabulary) + 1
        idx2word[len(vocabulary) + 1] = self.padding_text

        X = []
        for text in texts:
            words = text.split()
            x = []
            for w in words:
                if w in list(word2idx.keys()):
                    x.append(word2idx[w])

            lx = len(x)
            if lx >= self.max_features:
                x = x[:self.max_features]
            else:
                x = x + [word2idx[self.padding_text]] * (self.max_features - lx)

            X.append(x)

        return X, word2idx, idx2word
