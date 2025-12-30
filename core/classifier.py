from sklearn.svm import SVC
from joblib import dump, load

def train_model(X, y, model_path):
    model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    )
    model.fit(X, y)
    dump(model, model_path)
    return model

def load_model(model_path):
    return load(model_path)
