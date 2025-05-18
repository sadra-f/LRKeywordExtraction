from FeatureExtraction.Preprocess import clean_txt
from FeatureExtraction.POSTag import pos_tag, POS_TAG_REF
from FeatureExtraction.TFIDF import TFIDF
import pickle as pkl
import numpy as np

"""  This is honestly a mess it essentially extracts required features from the input text to be used in the logistic regression model  """

def text_to_numerical(txt):
    def nth_acc_term(index, n):
        try:
            while n != 0:
                if n < 0 :
                    index -= 1
                    if all_terms[index] in acc_inp_terms:
                        n += 1
                if n > 0:
                    index += 1
                    if all_terms[index] in acc_inp_terms:
                        n -= 1
            return index
        except IndexError:
            return -1

    txt = clean_txt(txt).lower()
    pos_tags = pos_tag([txt])
    with open("log/tfidf/termXidf.pkl", "rb") as f:
        log_idf = pkl.load(f)
    log_terms = list(log_idf.keys())
    all_terms = []
    for v in pos_tags[0]:
        all_terms.append(v[0])
    all_terms = list(set(all_terms))
    acc_inp_terms = set(all_terms)
    acc_inp_terms = list(acc_inp_terms.intersection(log_terms))
    tf = _tf(txt, acc_inp_terms)
    tfidf = {}
    for t in acc_inp_terms:
        tfidf[t] = tf[t] * log_idf[t]

    vectorized = []
    i = -1
    for k, t in enumerate(all_terms):
        if t not in acc_inp_terms:
            continue
        vectorized.append([t])
        i += 1
        try:
            if k < 2 or nth_acc_term(k, -2) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tfidf[all_terms[nth_acc_term(k, -2)]])

            if k < 1 or nth_acc_term(k, -1) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tfidf[all_terms[nth_acc_term(k, -1)]])

            vectorized[i].append(tfidf[t])

            if k >= len(all_terms) - 1 or nth_acc_term(k, 1) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tfidf[all_terms[nth_acc_term(k, 1)]])

            if k >= len(all_terms) - 2 or nth_acc_term(k, 2) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tfidf[all_terms[nth_acc_term(k, 2)]])

            if k < 2 or nth_acc_term(k, -2) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tf[all_terms[nth_acc_term(k, -2)]])

            if k < 1 or nth_acc_term(k, -1) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tf[all_terms[nth_acc_term(k, -1)]])

            vectorized[i].append(tf[t])

            if k >= len(all_terms) - 1 or nth_acc_term(k, 1) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tf[all_terms[nth_acc_term(k, 1)]])

            if k >= len(all_terms) - 2 or nth_acc_term(k, 2) == -1:
                vectorized[i].append(0)
            else:
                vectorized[i].append(tf[all_terms[nth_acc_term(k, 2)]])

            txt = txt.lower()

            vectorized[i].append(txt.find(t.lower()) / len(all_terms))
            vectorized[i].append(txt.rfind(t.lower())/ len(all_terms))

            vectorized[i].append(len(t))

            one_hots = np.zeros(len(POS_TAG_REF), dtype=int)
            one_hots[POS_TAG_REF.index(pos_tags[0][k][1])] = 1
            vectorized[i].extend(one_hots)

        except KeyError:
            vectorized.pop()
            i -= 1

    return vectorized


def _tf(txt, terms):
    tf = {}
    for t in terms:
        tf[t] = txt.lower().count(t.lower()) / len(terms)
    return tf