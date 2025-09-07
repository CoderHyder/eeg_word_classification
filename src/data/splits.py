import os
from pathlib import Path

def get_subject_id(filename: str) -> str:
    """Extract subject ID from filename (e.g., Data_Sample01_preprocessed.npz → Data_Sample01)."""
    return filename.replace("_preprocessed.npz", "")


def loso_split(train_dir):
    """
    Generate Leave-One-Subject-Out (LOSO) splits from the preprocessed training set.

    Args:
        train_dir (str): Path to train/ folder with *_preprocessed.npz files.

    Yields:
        (train_subjects, test_subject) for each fold
    """
    train_dir = Path(train_dir)
    files = sorted([f.name for f in train_dir.glob("*_preprocessed.npz")])
    if not files:
        raise FileNotFoundError(f"No preprocessed files found in {train_dir}")

    subject_ids = sorted(list(set(get_subject_id(f) for f in files)))

    for test_subj in subject_ids:
        train_subj = [s for s in subject_ids if s != test_subj]
        yield train_subj, test_subj

# Example usage
if __name__ == "__main__":
    train_dir = "data/preprocessed/train"
    for train_ids, test_id in loso_split(train_dir):
        print("Train:", train_ids, "| Test:", test_id)
