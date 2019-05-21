# SVD2vec

SVD2vec is a python library for representing documents words as vectors. Vectors are created using the PMI (Pointwise Mutual Information) and the SVD (Singular Value Decomposition).

This library implements recommendations from "Improving Distributional Similarity with Lessons Learned from Word Embeddings" (Omer Levy, Yoav Goldberg, and Ido Dagan). This papers suggests that traditional methods like PMI and SVD can be as good as word2vec by appling the same hyperparameters.

### Example

```shell
wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f
```


```python
>>> from svd2vec import svd2vec
>>> documents = [open("text8", "r").read().split(" ")]
>>> svd_vect = svd2vec(documents, window=2, min_count=100)

>>> svd_vect.analogy("paris", "france", "berlin")
# [('germany', 0.6977716641680641), ...]
>>> svd_vect.analogy("road", "cars", "rail")
# [('trains', 0.7532519174901262), ...]
>>> svd_vect.analogy("cow", "cows", "pig")
[('pigs', 0.6944101149919422), ...]

```

---

[Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00134) <br>
**Omer Levy**, **Yoav Goldberg**, and **Ido Dagan** <br>
Transactions of the Association for Computational Linguistics 2015 Vol. 3, 211-225 <br>
