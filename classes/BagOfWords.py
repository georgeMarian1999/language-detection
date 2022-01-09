import numpy as np

from utilities.utils import tokenize


class BagOfWords:

    def __init__(self):
        self.vocab = {}
        self.words = []

    def build_vocabulary(self, train_data):
        for paragraph in train_data:
            for word in tokenize(paragraph):
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
        return len(self.words)

    def get_features(self, data, number_sentences):
        result = np.zeros((number_sentences, len(self.words)))
        for idx, sentence in enumerate(data):
            for word in tokenize(sentence):
                if word in self.vocab:
                    result[idx, self.vocab[word]] += 1
        return result

