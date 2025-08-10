# Seq2Seq vs Transformer: Hindi Transliteration 

This project implements and compares two deep learning approaches — **Sequence-to-Sequence (Seq2Seq)** and **Transformer** — for Hindi transliteration using the **Dakshina dataset**. The task involves converting a Romanized Hindi string (Latin script) into its corresponding **Devanagari** script.

##  Project Overview
- **Dataset**: Dakshina dataset (Hindi)
- **Task**: Character-level transliteration from Latin to Devanagari script
- **Models**:
  - Encoder-Decoder (Seq2Seq) using GRU/LSTM
  - Transformer model
- **Optimization**: Hyperparameter tuning using **Weights & Biases (wandb) Sweeps**
- **Comparison**: Evaluated performance of the best Seq2Seq model vs. Transformer

##  Features
- Flexible model configurations
- Custom data preprocessing pipeline
- Support for batch normalization, dropout, and data augmentation
- Integrated training and evaluation scripts
- W&B integration for tracking experiments and hyperparameter sweeps

##  Dataset
The [Dakshina dataset](https://github.com/google-research-datasets/dakshina) contains Romanized text paired with its native script equivalent.  
For Hindi (`hi`):
- **Train**: `hi_train.csv`
- **Validation**: `hi_dev.csv`
- **Test**: `hi_test.csv`

## Installation
```bash
# Clone the repository
git clone https://github.com/Kahkashan2708/Hindi-Transliteration-Seq2Seq-vs-Transformer
cd Hindi-Transliteration-Seq2Seq-vs-Transformer

# Install dependencies
pip install -r requirements.txt

