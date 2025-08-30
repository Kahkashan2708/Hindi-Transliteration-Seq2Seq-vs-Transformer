# Seq2Seq vs Transformer on Dakshina Dataset

This repository explores and compares **Sequence-to-Sequence (Seq2Seq)** models (RNN, LSTM, GRU) and **Transformer architectures** for transliteration tasks using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina).  
The project also integrates **Weights & Biases (wandb)** for experiment tracking, hyperparameter tuning, and visualization.

---

##  Project Overview
The goal of this project is to build transliteration models that convert Romanized text (Latin script) into Indic scripts (such as Hindi).  

Two major approaches were implemented:
1. **Encoder–Decoder (Seq2Seq) Models**
   - Implemented with RNN, LSTM, and GRU.
   - Trained with attention mechanism.
   - Hyperparameter optimization performed using wandb sweeps.

2. **Transformer Model**
   - Implemented from scratch using PyTorch.
   - Multi-head attention and positional encoding included.
   - Tuned with wandb to select best-performing configuration.

---

##  Dataset
- **Dakshina dataset**: A multilingual dataset by Google Research.  
- Provides pairs of Romanized words and their corresponding native-script words.  
- For this project, we focused mainly on **Hindi transliteration**.  

---

## Models Implemented

### Seq2Seq (RNN/LSTM/GRU)
- Encoder: Processes Romanized input.
- Decoder: Generates Indic script output.
- Variants:
  - RNN Encoder–Decoder
  - LSTM Encoder–Decoder
  - GRU Encoder–Decoder
- Attention mechanism added for better alignment.
- Hyperparameters (embedding size, hidden size, learning rate, etc.) tuned via **wandb sweeps**.

###  Transformer
- Encoder: Multi-head self-attention + positional encoding.
- Decoder: Masked multi-head attention + encoder–decoder attention.
- Trained with teacher forcing and label smoothing.
- Hyperparameter tuning via wandb (layers, heads, embedding dimensions, dropout).

---

## Experiment Tracking with Weights & Biases
- All models integrated with **wandb** for:
  - Hyperparameter sweeps
  - Loss & accuracy visualization
  - Comparison of different architectures
  - Selection of the best-performing model  

---

##  Results & Comparison

| Model           | Strengths | Weaknesses | Performance (on Dakshina Test Set) |
|-----------------|-----------|------------|-------------------------------------|
| **RNN**         | Simple, lightweight | Struggles with long sequences | Moderate |
| **LSTM**        | Handles long dependencies better than RNN | Higher training cost | Better than RNN |
| **GRU**         | Efficient, faster than LSTM | Slightly less expressive than LSTM | Similar or better than LSTM |
| **Transformer** | Parallelized training, captures global context effectively | Needs more data & compute | **Best accuracy** |


## Final Results & Comparison

| Model            | Best Test Accuracy | Observation |
|------------------|--------------------|-------------|
| **Seq2Seq (LSTM)** | **38.34%**         | Performs reasonably well but struggles with longer dependencies and shows limited generalization. |
| **Transformer**   | **53.51%**         | Outperforms LSTM by a significant margin, better at capturing long-range dependencies and parallel training efficiency. |



##  Sample Predictions
| Input  | Predicted | Ground Truth |
|--------|-----------|--------------|
| ankit  | अंकित     | अंकित        |
| ankor  | अंकोर     | अंकोर        |
| angarak| अंगारक    | अंगारक       |


---

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Kahkashan2708/Seq2Seq-vs-Transformer.git
cd Seq2Seq-vs-Transformer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Seq2Seq-vs-Transformer.ipynb
```

### 4. Weights & Biases

```bash
wandb login
```
---

## Key Learnings

* Seq2Seq with attention performs reasonably well, but struggles with longer contexts.

* GRU gives a good balance between efficiency and accuracy compared to LSTM.

* Transformer models excel due to their ability to capture global dependencies and allow parallel computation.

* Wandb made it easy to compare models, tune hyperparameters, and visualize training.

## References
- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)  
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) 


