from sklearn.base import BaseEstimator
from joblib import load


def get_apple_model() -> tuple[BaseEstimator, str]:
    model: BaseEstimator = load('stocksense/res/models/apple_model.bin')
    with open('stocksense/res/html/apple_model.html') as html_file:
        html = html_file.read()
    return model, html
