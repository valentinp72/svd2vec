.. svd2vec documentation master file, created by
   sphinx-quickstart on Wed May 22 10:01:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to svd2vec's documentation!
===================================

**SVD2vec** is a python library for representing documents words as vectors.
Vectors are created using the **PMI** (Pointwise Mutual Information) and the 
**SVD** (Singular Value Decomposition).

This library implements recommendations from "Improving Distributional
Similarity with Lessons Learned from Word Embeddings" (Omer Levy, Yoav Goldberg,
and Ido Dagan) [#]_. This papers suggests that traditional methods like PMI and SVD
can be as good as word2vec by appling the same hyperparameters.

.. [#]
    `Improving Distributional Similarity with Lessons Learned from Word Embeddings <https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00134>`_.
    **Omer Levy**, **Yoav Goldberg**, and **Ido Dagan** 
    Transactions of the Association for Computational Linguistics 2015 Vol. 3, 211-225 

.. toctree::
   :maxdepth: 5 
   :caption: Contents:

   getting_started
   svd2vec

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
