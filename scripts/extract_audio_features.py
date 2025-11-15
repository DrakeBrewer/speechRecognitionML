from pathlib import Path
import numpy as np
import pandas as pd
import librosa


PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT_DIR / "data" / "audio" / "raw"
OUTPUT_PATH = PROJECT_ROOT_DIR / "data"/ "audio" / "processed" / "audio_features.csv"
SAMPLE_RATE = 22050
WINDOW_LEN = 1.0
HOP_LEN = 0.5
SILENCE_THRESHOLD = 0.01


def extract_features(file_path, speaker):
    print(f"Extracting features from {speaker}'s .wav file")

    mfcc_feat = librosa.feature.mfcc
    chroma_feat = librosa.feature.chroma_stft                                  # define librosa feature functions
    contrast_feat = librosa.feature.spectral_contrast
    zcr_feat = librosa.feature.zero_crossing_rate
    rms_feat = librosa.feature.rms
    mfcc_delta_feat = librosa.feature.delta

    raw_data, sampling_rate = librosa.load(file_path, sr=SAMPLE_RATE)          # raw waveform (amplitude/time)
    window_size = int(WINDOW_LEN * sampling_rate)                              # for processing small chunks at a time
    hop_size = int(HOP_LEN * sampling_rate)                                    # overlap
    feature_vectors = []                                                       # each windows (row) extracted numerical features

    frames = librosa.util.frame(raw_data, frame_length=window_size, hop_length=hop_size)
    for i in range(frames.shape[1]):
        window = frames[:,i]

        rms = rms_feat(y=window)
        if np.mean(rms) < SILENCE_THRESHOLD:                                   # skip over silent segments
            continue

        mfcc = mfcc_feat(y=window, sr=sampling_rate, n_mfcc=13)                # features to be extracted from each window
        chroma = chroma_feat(y=window, sr=sampling_rate)
        contrast = contrast_feat(y=window, sr=sampling_rate)
        zcr = zcr_feat(y=window)
        mfcc_delta = mfcc_delta_feat(mfcc)

        feature_list = [mfcc, chroma, contrast, zcr, rms, mfcc_delta]
        features = []
        for idx, feat in enumerate(feature_list):
            if idx == 3 or idx == 4:                                           # zcr & rms are single quantifiers (ok to append)
                features.append(np.mean(feat))
            else:
                features.extend(np.mean(feat, axis=1))                         # the rest are 2D -> must extend all values

        features.append(speaker)                                               # add speaker name to end
        feature_vectors.append(features)

    df = pd.DataFrame(feature_vectors)
    return df


def main():
    data_frames = [extract_features(file, file.stem) for file in AUDIO_DIR.glob("*.wav")]
    final_df = pd.concat(data_frames, ignore_index=True)                       # extract features from each wav file

    feature_names = (                                                          # create feature labels
    [f"mfcc_{i}" for i in range(1, 14)] +
    [f"chroma_{i}" for i in range(1, 13)] +
    [f"contrast_{i}" for i in range(1, 8)] +
    ["zcr", "rms"] +
    [f"mfcc_delta_{i}" for i in range(1, 14)] +
    ["speaker"]
    )

    final_df.columns = feature_names
    final_df.to_csv(OUTPUT_PATH, index=False)                                  # convert to csv
    print("Audio Data Successfully Processed")


if __name__ == '__main__':
    main()

