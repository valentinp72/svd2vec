# SVD2vec

SVD2vec is a python library for representing documents words as vectors. Vectors are created using the PMI (Pointwise Mutual Information) and the SVD (Singular Value Decomposition).

This library implements recommendations from "Improving Distributional Similarity with Lessons Learned from Word Embeddings" (Omer Levy, Yoav Goldberg, and Ido Dagan). This papers suggests that traditional methods like PMI and SVD can be as good as word2vec by appling the same hyperparameters.

Documentation can be found at [https://valentinp72.github.io/svd2vec/index.html](https://valentinp72.github.io/svd2vec/index.html)

### Example

```shell
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f
```

```python
# Building
>>> from svd2vec import svd2vec
>>> documents = [open("text8", "r").read().split(" ")]
>>> svd = svd2vec(documents, window=2, min_count=100)
```

```python
# I/O
>>> svd.save("svd.bin")
>>> svd = svd2vec.load("svd.bin")
```

```python
# Similarities
>>> svd.similarity("bad", "good")
# 0.4156516999158368
>>> svd.similarity("monday", "friday")
# 0.839529117681973
```

```python
# Most similar words
>>> svd.most_similar(positive=["january"], topn=2)
# [('february', 0.6854849518368631), ('october', 0.6653385092683669)]
>>> svd.most_similar(positive=['moscow', 'france'], negative=['paris'], topn=4)
# [('russia', 0.6221746629754187), ('ussr', 0.6024809889985986), ('soviet', 0.5794180517326273), ('bolsheviks', 0.5365123080505297)]
```

```python
# Analogies
>>> svd.analogy("paris", "france", "berlin")
# [('germany', 0.6977716641680641), ...]
>>> svd.analogy("road", "cars", "rail")
# [('trains', 0.7532519174901262), ...]
>>> svd.analogy("cow", "cows", "pig")
# [('pigs', 0.6944101149919422), ...]
>>> svd.analogy("man", "men", "woman")
# [('women', 0.7471792753875327), ...]
```

Using [Gensim](https://pypi.org/project/gensim/) you can load a `svd2vec` model using it's `word2vec` representation:
```python
>>> from gensim.models.keyedvectors import Word2VecKeyedVectors
>>> svd.save_word2vec_format("svd_word2vec_format.txt")
>>> keyed_vector = Word2VecKeyedVectors.load_word2vec_format("svd_word2vec_format.txt")
>>> keyed_vector.similarity("good", "bad")
# 0.54922897
```

---

[Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00134) <br>
**Omer Levy**, **Yoav Goldberg**, and **Ido Dagan** <br>
Transactions of the Association for Computational Linguistics 2015 Vol. 3, 211-225 <br>
