
import random
import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine

from collections import OrderedDict, Counter

class Utils:

    def flatten(lst):
        # Returns a flatten version of the given list
        # All sublists are merged onto a bigger list. Non-list elements are removed
        return [item for sublst in lst for item in sublst if isinstance(sublst, list)]

    def random_decision(probability):
        return random.random() < probability

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

    NRM_SCHEME_NONE   = "none"
    NRM_SCHEME_ROW    = "row"
    NRM_SCHEME_COLUMN = "column"
    NRM_SCHEME_BOTH   = "both"
    NRM_SCHEMES = [NRM_SCHEME_NONE, NRM_SCHEME_ROW, NRM_SCHEME_COLUMN, NRM_SCHEME_BOTH]

    def __init__(self,
                 documents,
                 size=150,
                 min_count=2,
                 window=10,
                 dyn_window_weight=WINDOW_WEIGHT_WORD2VEC,
                 cds_alpha=0.75,
                 neg_k_shift=5,
                 eig_p_weight=0,
                 nrm_type=NRM_SCHEME_ROW,
                 sub_threshold=1,  # 1e-5
                 workers=10):

        # -------------
        # args checking
        # -------------

        # dyn weight
        if dyn_window_weight == svd2vec.WINDOW_WEIGHT_HARMONIC:
            window_weighter = Utils.weight_harmonic
        elif dyn_window_weight == svd2vec.WINDOW_WEIGHT_WORD2VEC:
            window_weighter = Utils.weight_word2vec
        else:
            raise ValueError(dyn_window_weight + " not implemented as a weighter")

        # window type
        if isinstance(window, int):
            window = Utils.create_window(left=window, right=window, weighter=window_weighter)
        elif isinstance(window, tuple) and len(window) == 2 and all(map(lambda e: isinstance(e, int), window)):
            window = Utils.create_window(left=window[0], right=window[1], weighter=window_weighter)
        else:
            raise ValueError(window + " not implemented as a window yielder")

        # normalization type
        if nrm_type not in svd2vec.NRM_SCHEMES:
            raise ValueError(nrm_type + " cannot be used in as a normalization method")

        # -----------
        # args saving
        # -----------

        self.size          = size
        self.window        = window
        self.cds_alpha     = cds_alpha
        self.sub_threshold = sub_threshold
        self.neg_k_shift   = neg_k_shift
        self.eig_p_weight  = eig_p_weight
        self.nrm_type      = nrm_type

        # --------------------
        # document preparation
        # --------------------
        self.build_vocabulary(documents)
        self.subsampling()
        self.weighted_count_matrix = self.skipgram_weighted_count_matrix()

        # ---------------
        # pmi computation
        # ---------------
        self.pmi = self.sparse_pmi_matrix(self.sppmi_matrix(self.pmi_matrix()))
        # self.display_matrix(self.pmi)

        # ---------------
        # svd computation
        # ---------------

        self.svd_w, self.svd_c = self.svd()
        self.similarity("interieur", "bleu")
        self.similarity("bleu", "rouge")
        self.similarity("maison", "cheval")

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
        self.terms_counts   = Counter(self.all_words)

    def subsampling(self):
        new_docs = []
        for document in self.documents:
            new_words = []
            for word in document:
                word_frequency = 1.0 * self.terms_counts[word] / self.d_size
                prob = 1 - np.sqrt(self.sub_threshold / word_frequency)
                if not Utils.random_decision(prob):
                    # we keep the word
                    new_words.append(word)
            new_docs.append(new_words)
        self.build_vocabulary(new_docs)

    def skipgram_weighted_count_matrix(self):
        matrix = np.zeros((self.vocabulary_len, self.vocabulary_len))

        for document in self.documents:
            for word, context, weight in self.window(document):
                i_word    = self.vocabulary[word]
                i_context = self.vocabulary[context]
                matrix[i_word, i_context] += weight

        return matrix

    def pmi_matrix(self):
        # pointwise mutal information
        pmi = np.zeros((self.vocabulary_len, self.vocabulary_len))

        for word in self.vocabulary:
            for context in self.vocabulary:
                i_word    = self.vocabulary[word]
                i_context = self.vocabulary[context]
                pmi[i_word, i_context] = self.pmi(word, context)

        return pmi

    def ppmi_matrix(self, pmi):
        # positive pointwise mutal information
        zero = np.zeros(pmi.shape)
        return np.maximum(pmi, zero)

    def sppmi_matrix(self, pmi):
        # shifted positive pointwise mutal information
        spmi = pmi - np.log2(self.neg_k_shift)  # not sure if it's log or log2
        return self.ppmi_matrix(spmi)

    def sparse_pmi_matrix(self, pmi):
        sparsed = csc_matrix(pmi)
        return sparsed

    def svd(self):
        modified_k = min(self.size, self.pmi.shape[0] - 1)
        u, s, v = svds(self.pmi, k=modified_k)

        w_svd_p = u * np.power(s, self.eig_p_weight)
        c_svd   = v

        return w_svd_p, c_svd

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

        p_wc = n_wc / n_d
        p_w  = n_w  / n_d
        p_c  = np.power(n_c, self.cds_alpha) / np.power(n_d, self.cds_alpha)

        frac = p_wc / (p_w * p_c)

        if frac == 0:
            return -np.inf
        return np.log2(frac)

    def similarity(self, x, y):
        wx, cx = self.vectors(x)
        wy, cy = self.vectors(y)

        print(x, y)
        print(cosine(wx, wy))
        print(cosine(cx, cy))
        print("")
        #top = np.dot(wx, wy) + np.dot(cx, cy) + np.dot(wx, cy) + np.dot(cx, wy)
        #bot = (2 * np.sqrt(np.dot(wx, cx) + 1)) * (np.sqrt(np.dot(wy, cy) + 1))
        #print(top, bot)

    def vectors(self, word):
        if word in self.vocabulary:
            i_word = self.vocabulary[word]
            w = self.svd_w[i_word]
            c = self.svd_c[i_word]
            return w, c
        else:
            raise ValueError("Word " + word + " not in the vocabulary")

    #####
    # Debug
    #####

    def display_matrix(self, matrix, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.vocabulary
        z = {a: e for e, a in vocabulary.items()}
        v = [z[i] for i in sorted(vocabulary.values())]
        df = pd.DataFrame(matrix.toarray(), columns=v, index=v)
        df = df.applymap(lambda x: '{:4.2f}'.format(x) if x != 0 else "")
        print(df)
