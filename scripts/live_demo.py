import sounddevice as sd
import soundfile as sf
import numpy as np
from joblib import load
from pathlib import Path
from extract_audio_features import extract_features, get_feature_names

# load pickled model

def record_1sec():
    pass

def predict_one_instance(file_path):
    df = extract_features(file_path)
    # if df empty -> print("No speaker present")
    # else, add feature names, drop col ['speaker'], then model.predict(df)
    # print results
    pass

def main():
    while True:
        wav_file = record_1sec()
        predict_one_instance(wav_file)

if __name__ == '__main__':
    main()
