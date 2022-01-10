from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings
import seaborn as sns

from utilities.singletons import bow_model, model
from utilities.utils import read_data_csv, get_column_data, encode_languages, create_data_list

warnings.simplefilter("ignore")


def data_read(file):
    data = read_data_csv(file)
    text = get_column_data(data, "Text")
    languages = get_column_data(data, "Language")
    languages = encode_languages(languages)
    data_list = create_data_list(text)
    return text, languages, data_list


def create_model_unigram():
    text, languages, data_list = data_read(
        "/Users/georgemarianmihailescu/PycharmProjects/language-detection/train_data/language_detection.csv")
    bow_model.build_vocabulary(data_list)
    text = bow_model.get_features(data_list, len(data_list))
    text_train, text_test, languages_train, languages_test = train_test_split(text, languages, test_size=0.20)

    model.fit(text_train, languages_train)

    languages_prediction = model.predict(text_test)
    accuracy_sc = accuracy_score(languages_test, languages_prediction)
    # confusion_mat = confusion_matrix(languages_test, languages_prediction)
    # plt.figure(figsize=(15, 10))
    # sns.heatmap(confusion_mat, annot=True)
    plt.show()
    return accuracy_sc


def create_model_ngram(n):
    text, languages, data_list = data_read(
        "/Users/georgemarianmihailescu/PycharmProjects/language-detection/train_data/language_detection.csv")
    cv = CountVectorizer(ngram_range=(n, n))
    text = cv.fit_transform(data_list).toarray()
    text_train, text_test, languages_train, languages_test = train_test_split(text, languages, test_size=0.20)
    model.fit(text_train, languages_train)

    languages_prediction = model.predict(text_test)

    accuracy_sc = accuracy_score(languages_test, languages_prediction)
    print("Accuracy score is ", accuracy_sc)
    return accuracy_sc, cv
