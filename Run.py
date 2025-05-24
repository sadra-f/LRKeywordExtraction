from MlModel.LogisitcRegression import LogisticRegression as LR
import numpy as np
from FeatureExtraction.extractor import text_to_numerical
from MlModel.BERTBased import BERT_keyword
from FeatureExtraction.BERTbased import sb_vectorizer as sb
from Evaluation.helpers import cosine_similarity

# List of files the keywords of which are required
inp_files = ["test_input.txt", "test_input copy.txt"]

for file in inp_files:

    lr = LR()
    lr.load_model()

    with open(file, "r") as f:
        inp_str = f.read()
    # Extract keywords using KeyBERT
    KBERT_keys = BERT_keyword(inp_str)
    KBERT_keys = [val[0] for val in KBERT_keys]
    KBERT_key_sent = " ".join(KBERT_keys)

    features = np.array(text_to_numerical(inp_str))
    # remove the 'word' column from the dataset
    cleaned_features = np.array(features[:,1:], dtype=np.float64)

    predictions = lr.predict(cleaned_features)
    keyword_predictions = features[predictions][:,0]
    # put predicted words in a sentence together
    predict_key_sent = " ".join(keyword_predictions)
    #Load ground_truth keywords
    with open("test_ouput_keys.txt") as f:
        inp_gt = f.read()
    
    # Generate BERT embeddings from Custom and KeyBERT keywords
    embeddings = sb([KBERT_key_sent, predict_key_sent, inp_gt])
    # Print results
    print("\t- KeyBERT Predicted Keywords :\r\n", KBERT_key_sent, "\r\n")
    print("\t- Custom Predicted Keywords :\r\n", predict_key_sent)
    print("- Cosine Similarity of Custom and BERT keywords : ", cosine_similarity(embeddings[0], embeddings[1]))
    print("- Cosine Similarity of ground_truth and BERT keywords : ", cosine_similarity(embeddings[0], embeddings[2]))
    print("- Cosine Similarity of Custom and ground_truth keywords : ", cosine_similarity(embeddings[1], embeddings[2]))

print()