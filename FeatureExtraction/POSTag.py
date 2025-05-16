import nltk

# list of Part-of-Speech Tags that are used for words and not characters and punctuations
POS_TAG_REF = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN','JJ' ,'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 
               'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 
               'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

def pos_tag(docs:list[str]):
    """returns the POS tagged documents, removes stopwords after tagging if a list of stopwords is provided

    Args:
        docs (list[str]): documents to POS tag
        stopwords (list[str], optional): list of stopword strings. Defaults to None.
    """
    tokenized_list = [nltk.word_tokenize(doc) for doc in docs]
    return nltk.pos_tag_sents(tokenized_list)