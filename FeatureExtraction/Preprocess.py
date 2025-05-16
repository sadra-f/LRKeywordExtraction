import string

def remove_punctuation(txt:str):
    translator = str.maketrans('', '', string.punctuation)
    return txt.translate(translator)

def remove_numericals(txt:str):
    translator = str.maketrans('','', string.digits)
    return txt.translate(translator)

def clean_txt(txt):
    return remove_punctuation(remove_numericals(txt))

def is_bad_word(value:str):
    if len(value) < 3:
        return True
    for v in string.punctuation:
        if v in value:
            return True

    for v in string.digits:
        if v in value:
            return True
    return False

def remove_bad_words(words:list[str]):
    res = []
    for w in words:
        if len(w) < 3:
            continue
        for v in string.punctuation:
            if v in w:
                break
        else:
            for v in string.digits:
                if v in w:
                    break
            else:
                res.append(w)
    return res
