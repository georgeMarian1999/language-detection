import csv
import re

import pandas as pd
from sklearn import preprocessing

from utilities.conf import label_encoder


def read_data_csv(file):
    data = pd.read_csv(file)
    return data


def encode_languages(data):
    return label_encoder.fit_transform(data)


def decode_languages(data):
    return label_encoder.inverse_transform(data)


def clean_text(text):
    text = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
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


def write_to_csv(file, text, languagues, predicted):
    row_list = []
    with open(file, mode='w') as employee_file:
        row_list.append("Text")
        row_list.append("Actual Language")
        row_list.append("Predicted Language")
        employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(row_list)
        row_list.clear()

        for i in range(len(text)):
            row_list.clear()
            row_list.append(text[i])
            row_list.append(languagues[i])
            row_list.append(predicted[i])
            employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow(row_list)


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


def normalize_data(train_data, test_data, type=None):
    """
        Function reads a file, count number of sentences and extract ids of paragraphs and language codes.
        Return number of sentences, ids and codes extracted.
        :param  train_data: List<Integer> - the array of features for the train_data
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
