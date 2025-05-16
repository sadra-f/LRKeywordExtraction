import os
import numpy as np
import pickle as pkl
from pathlib import Path
import re
from FeatureExtraction.stopwords import STOP_WORDS

class TextKeyPair:
    def __init__(self, txt_path, key_path, txt, keys):
        self.txt_path = txt_path
        self.key_path = key_path
        if self.txt_path.name.split('.')[0] != self.key_path.name.split('.')[0]:
            raise ValueError("Text and Key Values are not from the same reference number.")
        self.id = txt_path.name.split(".")[0]
        self.txt = txt
        self.keys = []
        keys = re.split("\\n|,| ", keys)
        for key in keys:
            if key.strip() in txt and key.strip() != '' and key.lower().strip() not in STOP_WORDS:
                self.keys.append(key.strip())

    def __repr__(self):
        return f"{self.id}"

class Read:
    def __init__(self):
        pass


    def read_directory(path, file_pattern="*.txt", recursive=False, add_paths=False):
        """Reads all files with file_pattern in a directory

        Args:
            path (path_like): the path to the directory to read from.
            file_pattern (str, optional): Pattern of files to read. Defaults to "*.txt".
            recursive (bool, optional): Find files within directory and child directories if True. Defaults to False.
            add_paths (bool, optional): Return the Path to each read file as result in a tuple. Defaults to False.

        Returns:
            list[str]|list[tuple]: List of read files within the directory with/without their file path
        """
        if recursive:
            dir_files = [v for v in Path(path).rglob(file_pattern)]
        else:
            dir_files = [v for v in Path(path).glob(file_pattern)]
            
        return Read.read_multiple_files(dir_files, add_paths)

    def read_multiple_files(path_list, add_path:bool):
        res = []
        for p in path_list:
            res.append(Read.read_file(p, add_path))
        return res

    def read_file(path:Path, add_path:bool):
        res = None
        with open(path) as file:
            res = file.read()
        if add_path:
            return (path.absolute(), res)
        return res

    def NLM500_DS():
        PATH = "dataset/raw/NLM500/documents"
        txts = Read.read_directory(PATH, "*.txt", False, True)
        keys = Read.read_directory(PATH, "*.key", False, True)
        txts.sort(key=lambda x: int(x[0].name.split('.')[0]))
        keys.sort(key=lambda x: int(x[0].name.split('.')[0]))
        pairs = []
        for t, k in zip(txts, keys):
            pairs.append(TextKeyPair(t[0], k[0], t[1], k[1]))
        return pairs
    def read_pickle(path):
        res = None
        with open(path, "rb") as f:
            res = pkl.load(f)
        return res
    
    def read_numpy(path):
        return np.loadtxt(path)

class Write:
    def __init__(self):
        pass

    def write_file(path, content):
        Write.dir_assure(path)
        with open(path, "w") as f:
            f.write(content)

    def dir_exist(path):
        return os.path.exists(os.path.dirname(path))

    def dir_assure(path):
        if Write.dir_exist(path):
            return
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    def write_pickle(path, content):
        with open(path, "wb") as f:
            pkl.dump(content, f)

    def write_numpy(path, np_arr):
        np.savetxt(path, np_arr)