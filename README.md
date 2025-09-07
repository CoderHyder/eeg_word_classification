# EEG Word Classification

This project implements **EEG-based imagined speech classification** using the **FEIS dataset**, mapped to the **Emotiv Epoc X** headset configuration (14 channels, 128 Hz).

The pipeline includes preprocessing, multiple deep learning models (CNN, TCN, RNN), and evaluation under **Leave-One-Subject-Out (LOSO)** cross-validation.

---

## 📂 Project Structure

EEG-EpocX-Models/
│── README.md # Project documentation
│── requirements.txt # Dependencies
│── config.yaml # Global configs (default run)
│
├── data/
│ ├── raw/ # Original FEIS .mat files (64ch, 512Hz)
│ ├── preprocessed/ # Downsampled + EpocX channels (14ch, 128Hz)
│ └── splits/ # Train/validation/test subject splits
│
├── notebooks/ # Data exploration and visualization
├── src/ # Source code
│ ├── data/ # Dataset + preprocessing scripts
│ ├── models/ # Model definitions (CNN, TCN, RNN, MLP)
│ └── training/ # Train/eval loops, utilities
│
└── experiments/ # Model-specific YAML configs
├── exp1_cnn.yaml
├── exp2_tcn.yaml
└── exp3_rnn.yaml

yaml
Copy code

---

## ⚙️ Setup

1. **Create a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Prepare dataset:

Download the FEIS dataset from OSF or Kaggle.

Place subject .mat files in:

bash
Copy code
data/raw/FEIS/train/
data/raw/FEIS/validation/
🧹 Preprocessing
Convert FEIS (64ch, 512Hz) → EpocX (14ch, 128Hz):

bash
Copy code
python -m src.data.preprocess --raw_dir data/raw/FEIS/train --out_dir data/preprocessed/train
python -m src.data.preprocess --raw_dir data/raw/FEIS/validation --out_dir data/preprocessed/validation
This will:

Downsample from 512 Hz → 128 Hz

Select EpocX channels:

bash
Copy code
['AF3','F7','F3','FC5','T7','P7','O1',
 'O2','P8','T8','FC6','F4','F8','AF4']
Normalize each trial (zero mean, unit variance)

Save .npz files to data/preprocessed/

🏋️ Training
The project supports multiple architectures:

EEGNet-style CNN → experiments/exp1_cnn.yaml

Temporal Convolutional Network (TCN) → experiments/exp2_tcn.yaml

BiLSTM with Attention → experiments/exp3_rnn.yaml

Baseline MLP → src/models/baseline_mlp.py

Train with:

bash
Copy code
python -m src.training.train --config experiments/exp1_cnn.yaml
Features:

Leave-One-Subject-Out cross-validation (15 folds)

Early stopping on validation accuracy

Checkpointing of best model

TensorBoard logging

📊 Results
Outputs are saved in results/:

results/logs/ → TensorBoard logs

results/checkpoints/ → Model weights

results/figures/ → Confusion matrices, plots

View training progress:

bash
Copy code
tensorboard --logdir results/logs
```
