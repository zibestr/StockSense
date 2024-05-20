from sklearn.base import BaseEstimator
from joblib import load


def get_apple_model() -> BaseEstimator:
    model: BaseEstimator = load('stocksense/res/models/apple_model.bin')
    return model


def get_alphabet_model() -> BaseEstimator:
    model: BaseEstimator = load('stocksense/res/models/alphabet_model.bin')
    return model


def get_tesla_model() -> BaseEstimator:
    model: BaseEstimator = load('stocksense/res/models/tesla_model.bin')
    return model


def get_microsoft_model() -> BaseEstimator:
    model: BaseEstimator = load('stocksense/res/models/microsoft_model.bin')
    return model
