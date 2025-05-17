from keybert import KeyBERT

def BERT_keyword(inp):
    kb = KeyBERT()
    keys = kb.extract_keywords(inp)
    return keys