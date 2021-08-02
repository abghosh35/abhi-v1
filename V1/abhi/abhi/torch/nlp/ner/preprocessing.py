import numpy as np

class PreProcessSentences:
    """
    This module performs pre-processing of the raw texts and labels for NER exercise. Note that this works at WORD level.
    Inputs: 
        Sentences: List of raw texts of sentences
        Labels: List of raw tags corresponsing to the sentences
    Parameters:
        max_seq_len: Int (default=100) Maximum length of a sentence. If sentence is smaller than the max_seq_len, then '<PAD>' is added as new word to make the length=max_seq_len. If sentence is greater than the max_seq_len then only max_seq_len of words from first is considered.
        pad_text: string (default='<PAD>') Used to create padding of sentences or labels
    Attributes: 
        PaddedSentences: Return the padded (posterior) sentences making the max length as max_seq_len
        UniqueWords: List of unique words found in the sentences
        PaddedLabels: Return the padded (posterior) labels making the max length as max_seq_len
        UniqueLabels: List of unique words found in the sentences
        Word2Idx: Dictionary of word: index mapping
        Idx2Word: Dictionary of index: word mapping
        Tag2Idx: Dictionary of tag: index mapping
        Idx2Tag: Dictionary of index: tag mapping
        
        Sentences2Idx: Indexed version of the words for the sentences. Each word is replaced with corresponding index from Word2Idx
        Labels2Idx: Index version of each label in the Labels. Each label is replaced with corresponding index from Tag2Idx.
    """
    def __init__(self, Sentences, Labels, max_seq_len=100, pad_text='<PAD>'):
        self.Sentences = Sentences
        self.Labels = Labels
        self.max_seq_len = max_seq_len
        self.pad_text = pad_text
        
        self.PaddedSentences, self.UniqueWords = self.PadSentence(self.Sentences)
        self.PaddedLabels, self.UniqueLabels = self.PadLabels(self.Labels)
        
        self.Word2Idx = {w: i for i, w in enumerate(self.UniqueWords)}
        self.Idx2Word = {i: w for i, w in enumerate(self.UniqueWords)}
        
        self.Tag2Idx = {w: i for i, w in enumerate(self.UniqueLabels)}
        self.Idx2Tag = {i: w for i, w in enumerate(self.UniqueLabels)}
        
        self.Sentences2Idx = [[self.Word2Idx[w] for w in sent.split()] for sent in self.PaddedSentences]
        self.Labels2Idx = [[self.Tag2Idx[l] for l in Label] for Label in Labels]
        
        
    def PadSentence(self, Sentences):
        UniqueWords = []
        for i in range(len(Sentences)):
            _words_  = Sentences[i].split()
            UniqueWords.extend(_words_)
            n_words = len(_words_)
            
            if n_words < self.max_seq_len:
                _words_ = _words_ + [self.pad_text] * (self.max_seq_len - n_words)
            else:
                _words_ = _words_[:self.max_seq_len]
                
            Sentences[i] = " ".join(_words_)
            
        UniqueWords.append(self.pad_text)
            
        return Sentences, list(set(UniqueWords))
    
    def PadLabels(self, Labels):
        UniqueLabels = []
        for i in range(len(Labels)):
            UniqueLabels.extend(Labels[i])
            if len(Labels[i]) < self.max_seq_len:
                Labels[i] = Labels[i] + [self.pad_text] * (self.max_seq_len - len(Labels[i]))
            else:
                Labels[i] = Labels[i][:self.max_seq_len]
        
        UniqueLabels.append(self.pad_text)
        
        return Labels, list(set(UniqueLabels))