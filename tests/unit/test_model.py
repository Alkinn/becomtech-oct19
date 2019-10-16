import pytest
import joblib
import random
from mnist.model import Model


@pytest.fixture
def model_path():
    return "models/mnist_digits_clf.pkl"


@pytest.fixture
def data():
    X_test, y_test = joblib.load("data/X_test_and_y_test.pkl")
    return X_test, y_test


@pytest.fixture
def random_test_data():
    return [random.randint(0, 255) for i in range(784)]


@pytest.fixture
def prediction_range():
    return range(0, 10)


def test_class_exists():
    model = Model()
    assert model


def test_load_model(model_path):
    model = Model()
    model.load_model(model_path)
    assert model.clf


def test_from_model(model_path):
    model = Model.from_model(model_path)
    assert model and model.clf


def test_predict(model_path, random_test_data):
    model = Model.from_model(model_path)

    prediction = int(model.predict(random_test_data))

    assert prediction >= 0 and prediction <= 9 


