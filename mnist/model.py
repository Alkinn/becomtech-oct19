import pathlib
import joblib
from PIL import Image


class Model:

    def __init__(self):
        self.clf = None
    
    def load_model(self, model_path: pathlib.Path):
        self.clf = joblib.load(model_path)

    def predict(self, img_vect: list):
        if not isinstance(img_vect, list):
            raise TypeError
        result = self.clf.predict([img_vect])
        return result

    @classmethod
    def from_model(self, model_path: pathlib.Path):
        model = Model()
        model.load_model(model_path)
        return model