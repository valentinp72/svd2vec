import numpy as np
import pandas as pd

from collections import OrderedDict, Counter

class Utils:

    def flatten(lst):
        # Returns a flatten version of the given list
        # All sublists are merged onto a bigger list. Non-list elements are removed
        return [item for sublst in lst for item in sublst if isinstance(sublst, list)]

    def PMI(document, contexes, word, context, weight, alpha=0.75):
        frequency_word    = Utils.frequency(document, word)
        frequency_context = weight * Utils.frequency(document, context)
        frequency_word_context = Utils.frequency(contexes, (word, context), alpha=alpha)
        pmi = np.log(frequency_word_context / (frequency_word * frequency_context))
        return pmi

    def frequency(document, x, alpha=None):
        if isinstance(x, tuple):
            count = len([e for e in document if x[0] == e[0] and x[1] == e[1]])
        else:
            count = len([e for e in document if x == e])

        length = 1.0 * len(document)

        if alpha:
            count  = np.power(count,  alpha)
            length = np.power(length, alpha)
        return count / length

    def create_window(left, right, weighter):
        def window(document):
            doc_len = len(document)
            for iW, word in enumerate(document):
                for i in reversed(range(1, left)):
                    ictx = iW - i
                    if ictx <= 0:
                        break
                    ctx = document[ictx]
                    yield weighter(word, ctx, i, left)
                for i in range(1, right):
                    ictx = iW + i
                    if ictx >= doc_len:
                        break
                    ctx = document[ictx]
                    yield weighter(word, ctx, i, right)
        return window


    def PMI_matrix(window, vocabulary, document):
        n_vocabulary = len(vocabulary)
        m = np.zeros((n_vocabulary, n_vocabulary))
        for word, context, weight in window:
            pmi = Utils.PMI(document, window, word, context, weight)
            m[vocabulary[word]][vocabulary[context]] += pmi
        return m

    def weight_harmonic(word, context, dist, windowSize):
        weight = 1.0 / dist
        return (word, context, weight)

    def weight_word2vec(word, context, dist, windowSize):
        # weight = (1.0 * dist) / windowSize
        weight = 1.0 * windowSize / dist
        return (word, context, weight)

class svd2vec:

    WINDOW_WEIGHT_HARMONIC = 0
    WINDOW_WEIGHT_WORD2VEC = 1

    def __init__(self,
                 documents,
                 size=150,
                 min_count=2,
                 window=10,
                 window_weight=WINDOW_WEIGHT_WORD2VEC,
                 cds_alpha=0.75,
                 sub_threshold=1e-5,
                 workers=10):

        if window_weight == svd2vec.WINDOW_WEIGHT_HARMONIC:
            window_weighter = Utils.weight_harmonic
        elif window_weight == svd2vec.WINDOW_WEIGHT_WORD2VEC:
            window_weighter = Utils.weight_word2vec
        else:
            raise ValueError(window_weight + " not implemented as a weighter")

        if isinstance(window, int):
            window = Utils.create_window(left=window, right=window, weighter=window_weighter)
        elif isinstance(window, tuple) and len(window) == 2 and all(map(lambda e: isinstance(e, int), window)):
            window = Utils.create_window(left=window[0], right=window[1], weighter=window_weighter)
        else:
            raise ValueError(window + " not implemented as a window yielder")

        self.window = window
        self.build_vocabulary(documents)

        self.terms_counts = self.documents_counts_terms()
        self.weighted_count_matrix = self.skipgram_weighted_count_matrix()

        self.ppmi = self.ppmi_matrix()

        self.display_matrix(self.ppmi)

    #####
    # Building informations matrices and variables to be used later
    #####

    def build_vocabulary(self, documents):
        self.documents = documents
        self.all_words = Utils.flatten(self.documents)
        self.d_size    = len(self.all_words)

        unique = list(OrderedDict.fromkeys(self.all_words))
        self.vocabulary = OrderedDict()
        for i in range(len(unique)):
            self.vocabulary[unique[i]] = i

        self.vocabulary_len = len(self.vocabulary)

    def documents_counts_terms(self):
        counts = Counter(self.all_words)
        return counts

    def skipgram_weighted_count_matrix(self):
        matrix = np.zeros((self.vocabulary_len, self.vocabulary_len))

        for document in self.documents:
            for word, context, weight in self.window(document):
                i_word    = self.vocabulary[word]
                i_context = self.vocabulary[context]
                matrix[i_word, i_context] += weight

        return matrix

    def pmi_matrix(self):
        pmi = np.zeros((self.vocabulary_len, self.vocabulary_len))

        for word in self.vocabulary:
            for context in self.vocabulary:
                i_word    = self.vocabulary[word]
                i_context = self.vocabulary[context]
                pmi[i_word, i_context] = self.pmi(word, context)

        return pmi

    def ppmi_matrix(self):
        pmi  = self.pmi_matrix()
        zero = np.zeros(pmi.shape)
        return np.maximum(pmi, zero)

    #####
    # Getting informations
    #####

    def weight_count_term(self, term):
        count_term = self.terms_counts[term]
        return count_term

    def weight_count_term_term(self, t1, t2):
        i_t1 = self.vocabulary[t1]
        i_t2 = self.vocabulary[t2]
        weighted_count = self.weighted_count_matrix[i_t1, i_t2]
        return weighted_count

    def pmi(self, word, context):

        n_wc = self.weight_count_term_term(word, context)
        n_w  = self.weight_count_term(word)
        n_c  = self.weight_count_term(context)

        n_d = self.d_size

        frac = (n_wc * n_d) / (n_w * n_c)

        if frac == 0:
            return -np.inf
        return np.log2(frac)

    #####
    # Debug
    #####

    def display_matrix(self, matrix, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.vocabulary
        z = {a: e for e, a in vocabulary.items()}
        v = [z[i] for i in sorted(vocabulary.values())]
        df = pd.DataFrame(matrix, columns=v, index=v)
        df = df.applymap(lambda x: '{:4.2f}'.format(x) if x != 0 else "")
        print(df)
