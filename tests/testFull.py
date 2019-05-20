
import os
from context import svd2vec

from gensim.models import Word2Vec

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

svd_vectorized = svd2vec.svd2vec(
    documents,
    window=5,
    nrm_type=svd2vec.svd2vec.NRM_SCHEME_COLUMN,
    dyn_window_weight=svd2vec.svd2vec.WINDOW_WEIGHT_WORD2VEC,
    sub_threshold=1e-5)

svd_vectorized.display_similarity("signalisation", "signalisation")
svd_vectorized.display_similarity("signalisation", "pancarte")
svd_vectorized.display_similarity("et",            "le")
svd_vectorized.display_similarity("train",         "locomotive")
svd_vectorized.display_similarity("train",         "arrêté")
svd_vectorized.display_similarity("conducteur",    "sécurité")

svd_vectorized.display_most_similar(positive=["train"])
svd_vectorized.display_most_similar(positive=["conducteur"])
svd_vectorized.display_most_similar(positive=["un"])
svd_vectorized.display_most_similar(positive=["signalisation"])
svd_vectorized.display_most_similar(negative=["signalisation"])
svd_vectorized.display_most_similar(negative=["titulaire"])


word2vec_vectorized = Word2Vec(
    documents,
    size=150,
    window=5,
    min_count=1,
    workers=16
)

print("signalisation & signalisation", word2vec_vectorized.similarity("signalisation", "signalisation"))
print("train", word2vec_vectorized.most_similar(positive=["train"]))
