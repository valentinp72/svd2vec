
import os
from context import svd2vec

p = "/home/s150789/Ressources/output_linguistic/"

def file_get_col(file):
    splitted = [e.split(',') for e in file]
    choosed  = [a[1] if a != 'none' else a[0] for a in splitted]
    no_empty = [a for a in choosed if a != '']
    return no_empty

def all_files(files_paths):
    return [file_get_col(open(f).read().lower().splitlines()) for f in files_paths]

def list_files():
    return [p + f for f in os.listdir(p)]


documents = all_files(list_files()[0:50])

#documents = [open("/home/s150789/OP00575V_1_29012008.txt", "r").read().lower().splitlines()]

#print(documents)

svd2vec.svd2vec(documents, window=5, dyn_window_weight=svd2vec.svd2vec.WINDOW_WEIGHT_WORD2VEC, sub_threshold=1e-5)
