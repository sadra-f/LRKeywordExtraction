from Visualize.Dataset import plot
import pandas as pd
from MlModel.LogisitcRegression import LogisticRegression as LR
import numpy as np
from Evaluation.ClassificationEvaluators import evaluate_classification as ec
from FeatureExtraction.extractor import text_to_numerical
from MlModel.BERTBased import BERT_keyword

with open("test_input.txt", "r") as f:
    inp_str = f.read()

bert_keys = BERT_keyword(inp_str)
feature_vector = text_to_numerical(inp_str)
# remove the word column from the dataset
cleaned_feature_vector = [i[1:] for i in feature_vector]

lr = LR()
lr.load_weights()
keyword_predictions = lr.predict(cleaned_feature_vector)

# put predicted words in a sentence together
prediction_sent = ""
for i, val in enumerate(keyword_predictions):
    if val:
        prediction_sent += " " + feature_vector[i][0]
prediction_sent = prediction_sent.strip()

print()