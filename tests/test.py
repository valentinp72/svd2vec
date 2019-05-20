
from context import svd2vec


documents = [
    ['le', 'petit', 'maison', 'rouge', 'est', 'tres', 'grand', 'de', 'exterieur', 'mais', 'tres', 'petit', 'de', 'interieur'],
    ['le', 'petit', 'cheval', 'est', 'rouge', 'et', 'bleu'],
    ['i', 'am', 'very', 'confident', 'in', 'myself'],
]

print(documents)

svd_vectorized = svd2vec.svd2vec(
    documents,
    window=5,
    sub_threshold=1,
    dyn_window_weight=svd2vec.svd2vec.WINDOW_WEIGHT_WORD2VEC
)


svd_vectorized.display_similarity("interieur", "bleu")
svd_vectorized.display_similarity("bleu",      "rouge")
svd_vectorized.display_similarity("maison",    "cheval")
