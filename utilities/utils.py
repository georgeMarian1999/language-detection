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
    with open(file, mode='w', encoding="utf-8") as employee_file:
        row_list.append("Text")
        row_list.append("Actual Language")
        row_list.append("Predicted Language")
        wr = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(row_list)
        row_list.clear()

        for i in range(len(text)):
            row_list.clear()
            row_list.append(text[i])
            row_list.append(languagues[i])
            row_list.append(predicted[i])
            wr = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(row_list)

def write_accuracy_to_csv(file, languages_accuracy):
    row_list = []
    with open(file, mode='w', encoding="utf-8") as employee_file:
        row_list.append("Language")
        row_list.append("Accuracy")
        wr = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(row_list)
        row_list.clear()

        for language in languages_accuracy.keys():
            row_list.clear()
            row_list.append(language)
            row_list.append(languages_accuracy[language])
            wr = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(row_list)


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

