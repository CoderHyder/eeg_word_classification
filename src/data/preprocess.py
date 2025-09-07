import os
import argparse
import numpy as np
import scipy.io as sio
from scipy.signal import decimate
from tqdm import tqdm

# Epoc X channels
EPOCX_CHANNELS = [
    'AF3','F7','F3','FC5','T7','P7','O1',
    'O2','P8','T8','FC6','F4','F8','AF4'
]

def preprocess_subject(file_path, out_dir, fs_target=128):
    """Load FEIS .mat file, downsample to target Hz, select Epoc X channels, save as .npz"""
    mat = sio.loadmat(file_path)

    # find correct struct
    epo = None
    for key in ["epo_train", "epo_validation", "epo_test", "epo"]:
        if key in mat:
            epo = mat[key]
            break
    if epo is None:
        raise ValueError(f"No epo struct found in {file_path}")

    # extract
    X = epo["x"][0, 0]   # (time, channels, trials)
    y = epo["y"][0, 0]   # (classes, trials)
    fs = int(epo["fs"][0, 0][0, 0])
    ch_names = [str(c[0]) for c in epo["clab"][0, 0][0]]

    # rearrange -> (trials, channels, time)
    X = np.transpose(X, (2,1,0))
    labels = np.argmax(y, axis=0)  # shape: (n_trials,)

    # downsample safely
    factor = int(fs / fs_target)
    X_ds = []
    min_len = None

    for i in range(X.shape[0]):  # trials
        trial_ds = []
        for j in range(X.shape[1]):  # channels
            sig_ds = decimate(X[i, j, :], factor)
            trial_ds.append(sig_ds)
            if min_len is None or len(sig_ds) < min_len:
                min_len = len(sig_ds)
        X_ds.append(trial_ds)

    # crop all signals to same length
    X_out = np.zeros((X.shape[0], X.shape[1], min_len))
    for i, trial in enumerate(X_ds):
        for j, sig in enumerate(trial):
            X_out[i, j, :] = sig[:min_len]

        # select EpocX channels
    idx = [ch_names.index(ch) for ch in EPOCX_CHANNELS if ch in ch_names]
    X_out = X_out[:, idx, :]

    # --- Normalize each trial ---
    X_out = (X_out - X_out.mean(axis=-1, keepdims=True)) / (X_out.std(axis=-1, keepdims=True) + 1e-6)

    # save
    subj_id = os.path.basename(file_path).split('.')[0]
    out_path = os.path.join(out_dir, f"{subj_id}_preprocessed.npz")
    np.savez(out_path, X=X_out, y=labels)
    print(f"Saved {out_path}: {X_out.shape} | labels: {labels.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Folder containing train/ and validation/ subfolders")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Folder to save preprocessed data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ["train", "validation"]:
        split_raw = os.path.join(args.raw_dir, split)
        split_out = os.path.join(args.out_dir, split)
        os.makedirs(split_out, exist_ok=True)

        mat_files = [f for f in os.listdir(split_raw) if f.endswith(".mat")]
        print(f"Found {len(mat_files)} files in {split_raw}")

        for f in tqdm(mat_files, desc=f"Preprocessing {split}"):
            file_path = os.path.join(split_raw, f)
            preprocess_subject(file_path, split_out)
