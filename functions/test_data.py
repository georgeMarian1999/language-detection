from functions.detect_language import detect_language, detect_language_ngram
from utilities.utils import read_data_csv, get_column_data


def test_data(file):
    features = {}
    data = read_data_csv(file)
    all_sentences = get_column_data(data, "Text")
    languages = get_column_data(data, "Language")
    for lang in languages:
        features[lang] = 0
    wrong = 0.0
    predicted = []
    for i in range(0, len(all_sentences)):
        predicted.append(detect_language(all_sentences[i]))
        if languages[i] != predicted[i]:
            features[languages[i]] += 1
            wrong += 1
    accuracy = 1 - wrong / len(all_sentences)
    return all_sentences, languages, predicted, accuracy, features


def test_data_ngram(file, cv):
    features = {}
    data = read_data_csv(file)
    all_sentences = get_column_data(data, "Text")
    languages = get_column_data(data, "Language")
    for lang in languages:
        features[lang] = 0
    wrong = 0.0
    predicted = []
    for i in range(0, len(all_sentences)):
        predicted.append(detect_language_ngram(all_sentences[i], cv))
        if languages[i] != predicted[i]:
            features[languages[i]] += 1
            wrong += 1
    accuracy = 1 - wrong / len(all_sentences)
    return all_sentences, languages, predicted, accuracy, features


def get_feature_accuracy(file, features):
    number_of_language = {}
    result = {}
    lang = []
    data = read_data_csv(file)
    languages = get_column_data(data, "Actual Language")
    for l in languages:
        lang.append(l)

    for language in features.keys():
        number_of_language[language] = lang.count(language)
    for l in number_of_language.keys():
        accuracy = 1 - features[l] / number_of_language[l]
        result[l] = accuracy
    return result
