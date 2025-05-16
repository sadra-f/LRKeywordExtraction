import nltk
import time
import itertools
import numpy as np
from FeatureExtraction.stopwords import *
from FeatureExtraction.Preprocess import clean_txt
from IO.RW import Read , Write

class TFIDF:
    def __init__ (self, stopwords=STOP_WORDS):
        self.stopwords = stopwords
        self.docs = None
        self.docs_concat = None
        self.docs_tokenize = None
        self.terms_docs = None
        self.terms = None
        self.tf = None
        self.idf = None
        self.tfidf = None
        self.log_paths = {
            "tf": "log/tfidf/tf",
            "idf": "log/tfidf/idf",
            "tfidf": "log/tfidf/tfidf",
            "terms_docs" : "log/tfidf/termXdoc.pkl",
            "terms" : "log/tfidf/terms.pkl"
            }

    def extract(self, docs:list[str]):
        for i, doc in enumerate(docs):
            docs[i] = clean_txt(doc)
        self.docs = docs
        self.docs_concat = "\r\n".join(self.docs)
        self.terms_docs = TFIDF.extract_terms(self.docs, False)
        self.terms = list(set(itertools.chain.from_iterable(self.terms_docs)))
        self.terms.sort()
        self.tf = TFIDF.calc_tf(self.docs, self.terms)
        self.idf = TFIDF.calc_idf(self.docs, self.terms)
        self.tfidf =  self.tf * self.idf.reshape((len(self.idf), 1))
        return self
    
    def extract_terms(doc:str|list[str], unique=True, stopwords=STOP_WORDS, lang="english"):
        if type(doc) is list:
            terms = []
            for d in doc:
                tmp_trms = [v.lower() for v in nltk.word_tokenize(d, lang) if not v.lower() in stopwords]
                if unique:
                    tmp_trms = list(set(tmp_trms))
                tmp_trms.sort()
                terms.append(tmp_trms)
        else:
            terms = [v.lower() for v in nltk.word_tokenize(doc, lang) if not v.lower() in stopwords]
            if unique:
                terms = list(set(terms))
            terms.sort()
        return terms

    def calc_tf(separated_terms:list[list[str]], terms:list[str]):
        tf = np.ones((len(terms), len(separated_terms)))
        doc_count = len(separated_terms)
        for i, term in enumerate(terms):
            for d_t in separated_terms:
                tf[i] += ( d_t.count(term) / len(d_t))
        #log10 here is to normalize the values
        tf = np.log10(np.array(tf))
        return tf
    
    def calc_idf(separated_terms, terms):
        idf = []
        for i, trm in enumerate(terms):
            idf.append(1)
            for doc_trms in separated_terms:
                idf[i] += 1 if trm in doc_trms else 0
            
        idf = np.log10(len(separated_terms) / np.array(idf))
        return idf
    
    def _load_from_log_(self):
        terms_docs = Read.read_pickle(self.log_paths["terms_docs"])
        terms = Read.read_pickle(self.log_paths["terms"])
        tf = Read.read_numpy(self.log_paths["tf"])
        idf = Read.read_numpy(self.log_paths["idf"])
        tfidf = Read.read_numpy(self.log_paths["tfidf"])
        
        self.terms_docs = terms_docs
        self.terms = terms
        self.tf = tf
        self.idf = idf
        self.tfidf = tfidf
        
        return self

    def _save_to_log_(self):
        Write.write_pickle(self.log_paths["terms_docs"], self.terms_docs)
        Write.write_pickle(self.log_paths["terms"], self.terms)
        Write.write_numpy(self.log_paths["tf"], self.tf)
        Write.write_numpy(self.log_paths["idf"], self.idf)
        Write.write_numpy(self.log_paths["tfidf"], self.tfidf)
