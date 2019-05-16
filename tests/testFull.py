
from context import svd2vec


documents = [open("/home/s150789/OP00575V_1_29012008.txt", "r").read().lower().splitlines()]

print(documents)

svd2vec.svd2vec(documents, window=5, dyn_window_weight=svd2vec.svd2vec.WINDOW_WEIGHT_WORD2VEC)
