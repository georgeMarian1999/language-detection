from functions.detect_language import detect_language
from utilities.utils import read_data_csv, get_column_data, create_data_list


def test_data(file):
    data = read_data_csv(file)
    all_sentences = get_column_data(data, "Text")
    languages = get_column_data(data, "Language")
    wrong = 0.0
    predicted = []
    for i in range(0, len(all_sentences)):
        predicted.append(detect_language(all_sentences[i]))
        if languages[i] != predicted[i]:
            wrong += 1
    accuracy = 1 - wrong/len(all_sentences)
    return all_sentences, languages, predicted, accuracy

