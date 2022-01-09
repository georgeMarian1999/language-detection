import re

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def read_data_csv(file):
    data = pd.read_csv(file)
    return data


def encode_languages(data):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(data)


def clean_text(text):
    text = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', text)
    # text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    return text


def create_data_list(texts):
    data_list = []
    for text in texts:
        text = clean_text(text)
        data_list.append(text)
    return data_list


def get_column_data(data, column):
    return data[column]


def word_extraction(sentence):
    """
        Function extracts words of a text and removes special characters and numbers.
        Return a list of extracted words.
        :param sentence: string
        :return: cleaned_text - List<string>
    """
    words = re.sub("[^\w]", " ", sentence).split()
    # words = sentence.split()
    cleaned_text = []
    for w in words:
        if (w.isnumeric()) == False:
            if len(w) > 1:
                if w.isalpha():
                    cleaned_text.append(w.lower())

    return cleaned_text


def tokenize(sentences):
    """
        Function extracts words of a text and removes duplicates and one letter words.
        Return a list of words alphabetically sorted.
        :param sentences: string
        :return: words - List<string>
    """
    words = []
    wi = word_extraction(sentences)
    words.extend(wi)
    words = sorted(list(set(words)))
    return words


# str = "marian george nu prea stie stie python. Www  weee w w w www  wd ew  w wwfdwdfww."
# str2 =  "marian stie george 112 nu prea  stie @ //  $#python ## %%."
# wordss = tokenize(str)
# print("tokenize: ", wordss)
# cleaned_text = word_extraction(str2)
# print("word_extaction: ", cleaned_text)

def read_sentences(file):
    """
        Function reads a file and count number of sentences.
        Return a list of sentences and number of sentences.
        :param file: string - File name
        :return: number_sentences - Integer
                 text - List<string>
    """
    number_sentences = 0
    text = []
    fil = open(file, "r", encoding="utf-8")
    for sentences in fil:
        number_sentences += 1
        text.append(sentences)

    return number_sentences, text


# print("numbe1 sent: ", number_sentences)
# print("numbe1 trainingdata: ", training_data)
# print("numbe2 sent: ", number_sentences2)
# print("numbe2 testinggdata: ", testing_data)


def test(file):
    """
        Function reads a file, count number of sentences and extract ids of paragraphs and language codes.
        Return number of sentences, ids and codes extracted.
        :param file: string - File name
        :return: number_sentences - Integer
                 ids - List<string> - ids of paragraphs
                 language_codes - List<string>
    """
    number_sentences = 0
    language_codes = []
    ids = []

    with open(file, 'r+') as file:
        for sentences in file:
            i = sentences.split(None, 1)
            idx = i[0]
            ids.append(idx)
            text = int(i[1])
            number_sentences += 1
            language_codes.append(text)
    return number_sentences, ids, language_codes


def normalize_data(train_data, test_data, type=None):
    """
        Function reads a file, count number of sentences and extract ids of paragraphs and language codes.
        Return number of sentences, ids and codes extracted.
        :param  train_data: List<Integer> - the array of features for the data
                test_data: List<Integer> - the array of the known languages
        :return: number_sentences - Integer
                ids - List<string> - ids of paragraphs
                     language_codes - List<string>
    """
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1' or type == 'l2':
        scaler = preprocessing.Normalizer(norm=type)

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_test_data
    else:
        return train_data, test_data
