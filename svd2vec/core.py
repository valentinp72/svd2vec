import numpy as np
import pandas as pd

class Utils:

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
        print(vocabulary)
        m = np.zeros((n_vocabulary, n_vocabulary))
        for word, context, weight in window:
            pmi = Utils.PMI(document, window, word, context, weight)
            m[vocabulary[word]][vocabulary[context]] += weight * pmi
        Utils.display_matrix(m, vocabulary)
        return m

    def display_matrix(matrix, vocabulary):
        z = {a: e for e, a in vocabulary.items()}
        v = [z[i] for i in sorted(vocabulary.values())]
        df = pd.DataFrame(matrix, columns=v, index=v)
        print(df)

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
                 window=10,
                 window_weight=WINDOW_WEIGHT_WORD2VEC,
                 min_count=2,
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


        self.vocabulary = self.get_vocabulary(documents)

        for document in documents:
            win = [w for w in window(document)]
            m = Utils.PMI_matrix(win, self.vocabulary, document)
            print(m)
            input("a")
            ctxs = [ctx for ctx in window(document)]
            print(ctxs)
            for word, context, weight in ctxs:
                pmi = Utils.PMI(document, ctxs, word, context, weight)
                print(word, context, pmi)

    def get_vocabulary(self, documents):
        all = [e for document in documents for e in document]
        unique = set(all)
        return {word: i for i, word in enumerate(unique)}
