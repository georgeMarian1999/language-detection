from algorithm.LanguagePredict import create_model_unigram, create_model_ngram
from functions.test_data import test_data, get_feature_accuracy, test_data_ngram
from utilities.utils import write_to_csv, write_accuracy_to_csv


def run_unigram():
    create_model_unigram()
    text, languages, predicted, accuracy, features = test_data("test_data/test_data_text.csv")
    write_to_csv("result/result_test_data.csv", text, languages, predicted)
    languages_accuracy = get_feature_accuracy("result/result_test_data.csv", features)
    write_accuracy_to_csv("result/language_accuracy.csv", languages_accuracy)
    print("Accuracy of detecting from test data is ", accuracy)


def run_ngram():
    accuracy_sc, cv = create_model_ngram(2)
    text, languages, predicted, accuracy, features = test_data_ngram("test_data/test_data_text.csv", cv)
    write_to_csv("result/result_test_data_2gram.csv", text, languages, predicted)
    languages_accuracy = get_feature_accuracy("result/result_test_data_2gram.csv", features)
    write_accuracy_to_csv("result/language_accuracy_2gram.csv", languages_accuracy)
    print("Accuracy of detecting from test data is ", accuracy)


def main():
    run_ngram()


if __name__ == '__main__':
    main()
