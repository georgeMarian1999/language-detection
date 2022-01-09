from algorithm.LanguagePredict import create_model, svm_model
from functions.test_data import test_data
from utilities.utils import write_to_csv


def main():
    # create_model()
    svm_model()
    text, languages, predicted, accuracy = test_data("test_data/test_data_text.csv")
    write_to_csv("result/result_test_data.csv", text, languages, predicted)
    print("Accuracy of detecting from test data is ", accuracy)


if __name__ == '__main__':
    main()
