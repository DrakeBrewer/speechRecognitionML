import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import seaborn as sns
from joblib import dump

RANDOM_STATE = 42

def build_features():
    pass

def train():
    df = pd.read_csv("../../../data/audio/processed/audio_features.csv")
    
    # slplit data into tain/testing + encode speaker names
    x = df.drop(columns=["speaker", "session"], axis=1)
    y_encoded = df["speaker"].map({"Drake":0, "Melissa":1, "Lisa":2, "Dan":3, "David":4})
    groups = df["session"]

    # auto target encoding?
    # label_encoder = LabelEncoder()
    # y_encoded2 = label_encoder.fit_transform(y)
    # print("Classes:", label_encoder.classes_)

    gkfolds = GroupKFold(n_splits=5)

    gb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(mutual_info_classif, k="all")),
    ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    gb_grid = {
        "selector__k": [20,30,"all"],
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 5]
    }

    gb_search = GridSearchCV(gb_pipe, gb_grid, scoring="f1_macro", cv=gkfolds, n_jobs=-1)
    gb_search.fit(x, y_encoded, groups=groups)
    print("Gradient Boosting:")
    print(f"Best F1 Score: {gb_search.best_score_}")
    print(f"Best Params: {gb_search.best_params_}")

    # SVC Classifier
    svc_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(mutual_info_classif, k="all")),
        ('classifier', SVC(probability=True))
    ])

    svc_grid = {
        "selector__k": [30, 40, "all"],
        "classifier__C": [10, 25, 50],
        "classifier__gamma": ["scale", "auto", 0.1, 0.01],
        "classifier__kernel": ["rbf", "poly"],
        "classifier__degree": [2, 3]
    }

    svc_search = GridSearchCV(svc_pipe, svc_grid, scoring="f1_macro", cv=gkfolds, n_jobs=-1)
    svc_search.fit(x, y_encoded, groups=groups)
    print("SVC:")
    print(f"Best F1: {svc_search.best_score_}")
    print(f"Best Params: {svc_search.best_params_}")

    # best model
    model = svc_search.best_estimator_
    colum_names = x.columns.to_list()
    print("Model Successfully Trained")
    return gb_model

def predict():
    pass

def save(model):
    dump(model, "speech_rec_model.pkl")

    # need session_id as well... ?
    # bundle = {
    #     "model": gb_model, 
    #     "col_names": colum_names, 
    #     "label_encoder": Label_encoder
    # }
