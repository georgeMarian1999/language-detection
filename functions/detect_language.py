from utilities.singletons import bow_model, model
from utilities.utils import decode_languages


def detect_language(text):
    text_data = bow_model.get_features([text], 1)
    lang = model.predict(text_data)
    lang = decode_languages(lang)
    return lang[0]
