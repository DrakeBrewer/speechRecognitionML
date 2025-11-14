import os
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = "audio_data"              # our wav files
OUTPUT_CSV = "audio_features.csv"
SAMPLE_RATE = 22050
WINDOW_DURATION = 1.0   			  # 1 sec clip
HOP_DURATION = 0.5      			  # overlap 50%
SILENCE_THRESHOLD = 0.01  			  # skip quiet clips

def extract_Features(file_path, speaker):

    # extract amplitude (raw) data from each audio file
    raw_data, sampling_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    window_size = int(WINDOW_DURATION * sampling_rate)                  # for processing small chunks at a time
    hop_size = int(HOP_DURATION * sampling_rate)                        # overlap

    feature_vectors = []                                                # holds each window's extracted features
    for i in range(0, len(raw_data) - window_size, hop_size):
        window = raw_data[i : i + window_size]

        # skip over silent segments
        rms = np.mean(librosa.feature.rms(y=window))
        if rms < SILENCE_THRESHOLD:
            continue

        # extract features from window's raw data
        mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=window, sr=sampling_rate)
        contrast = librosa.feature.spectral_contrast(y=window, sr=sampling_rate)
        zcr = librosa.feature.zero_crossing_rate(y=window)
        rms = librosa.feature.rms(y=window)
        mfcc_delta = librosa.feature.delta(mfcc)

        # combine into one feature_vector (2D -> 1D vector)
        feature_list = [mfcc, chroma, contrast, zcr, rms, mfcc_delta]
        features = []
        for feat in feature_list:
            features.append(np.mean(feat, axis=1))
        feature_vectors.append(mean_features)

    # create feature labels for data frame (col names)
    feature_names = []
    feature_names.append([f"mfcc_{i}" for i in range(1, 14)])
    feature_names.append([f"chroma_{i}" for i in range(1, 13)])
    feature_names.append([f"contrast_{i}" for i in range(1, 8)])
    feature_names.append(["zcr", "rms"])
    feature_names.append([f"mfcc_delta_{i}" for i in range(1, 14)])

    # create && return data frame
    df = pd.DataFrame(feature_vectors, columns=feature_names)
    df["speaker"] = speaker
    return df

def main():
    data_frames = []
    for file in os.listdir(AUDIO_DIR):
        if file.endsith("wav"):
            path = os.path.join(AUDIO_DIR, file)
            df = extract_Features(path, os.path.splitext(file[0]))
            data_frames.append(df)

    final_df = pd.concat(data_frames, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Audio Data Successfully Converted to CSV")

if __name__ == '__main__':
    main()

