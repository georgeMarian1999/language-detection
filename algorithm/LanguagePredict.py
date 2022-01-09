from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from classes.BagOfWords import BagOfWords
from utilities.utils import read_data_csv, get_column_data, encode_languages, create_data_list

warnings.simplefilter("ignore")


def data_read(file):
    data = read_data_csv(file)
    text = get_column_data(data, "Text")
    languages = get_column_data(data, "Language")
    languages = encode_languages(languages)
    data_list = create_data_list(text)
    bow_model = BagOfWords()
    bow_model.build_vocabulary(data_list)
    text = bow_model.get_features(data_list, len(data_list))

    text_train, text_test, languages_train, languages_test = train_test_split(text, languages, test_size=0.20)

    model = MultinomialNB()
    model.fit(text_train, languages_train)

    languages_prediction = model.predict(text_test)
    accuracy_sc = accuracy_score(languages_test, languages_prediction)
    # confusion_mat = confusion_matrix(languages_test, languages_prediction)

    print("Accuracy is :", accuracy_sc)

    # plt.figure(figsize=(15, 10))
    # sns.heatmap(confusion_mat, annot=True)
    # plt.show()
