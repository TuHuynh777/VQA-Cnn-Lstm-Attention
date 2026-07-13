# Visual Question Answering (VQA) System with CNN-LSTM and Attention Mechanism

## Overview
This project implements a **Visual Question Answering (VQA)** system that answers natural language questions about images focused on **human activities and emotions**. The system combines **Computer Vision (ResNet50)** and **Natural Language Processing (LSTM)** to understand both an image and a question, then generate a relevant answer.

Four model variants were built and compared to study the effect of transfer learning and attention mechanisms:

| Model | CNN Backbone | Attention |
|---|---|---|
| PretrainedNoAttention | Pretrained ResNet50 (ImageNet) | No |
| PretrainedWithAttention | Pretrained ResNet50 (ImageNet) | Yes |
| TrainedNoAttention | ResNet50 trained from scratch | No |
| TrainedWithAttention | ResNet50 trained from scratch | Yes |

## Problem Statement
Visual Question Answering requires a model to jointly interpret an image and a natural language question to produce a correct, context-aware answer (e.g., Q: "What is the man doing?" A: "He is sitting on a chair and drinking tea."). This project restricts the domain to human activity and emotion recognition to keep the dataset focused and answers consistent.

## Tech Stack
- **Python 3**, **PyTorch**
- **ResNet50** (CNN) for image feature extraction
- **LSTM** (Encoder-Decoder) for question encoding and answer generation
- **Attention Mechanism** for dynamic image-region focus
- **Beam Search & Greedy Decoding** for inference
- Data source: Kaggle (custom-curated human activity images), Q&A pairs generated with an LLM (Grok)

## Dataset & Preprocessing
- Images filtered to clear, non-blurry photos showing human activities/emotions.
- Questions restricted to activity identification and emotional state.
- Text pipeline: tokenization, vocabulary construction (`<pad>`, `<start>`, `<end>`, `<unk>`), sequence padding.
- Image pipeline: resize/crop to 224x224, ImageNet normalization, light augmentation (ColorJitter) for training.
- Train/validation split: 80/20.

## Model Architecture
1. **Image Encoder:** ResNet50 extracts either a global 2048-d feature vector (no-attention variants) or a 7x7x2048 spatial feature map (attention variants).
2. **Question Encoder:** LSTM encodes the tokenized question into a fixed-size hidden representation.
3. **Fusion:** Image and question features are concatenated/attended and used to initialize the decoder LSTM's hidden and cell states.
4. **Answer Decoder:** LSTM generates the answer word-by-word, using either greedy decoding or beam search at inference time.
5. **Attention (optional):** Computes per-region attention weights at each decoding step so the model can focus on relevant image regions.

## Training Details
- Loss: Cross-Entropy (ignoring `<pad>` tokens)
- Optimizer: Adam (lr = 0.001, with weight decay)
- Early stopping: patience = 5 epochs based on validation loss
- Max epochs: 40

## Evaluation
Models were compared using:
- Training loss curve
- Validation loss curve
- Validation accuracy curve

Results showed that attention-based models achieved better validation accuracy and more stable convergence than non-attention counterparts, showing the benefit of dynamic image-region focus for this task.

## Repository Structure
```
vqa-cnn-lstm-attention/
├── notebooks/
│   ├── Code_Midterm_DL.ipynb          # Data preprocessing, model definitions, training loop
│   ├── beam_search_inference.ipynb    # Beam search decoding for all 4 models
│   └── greedy_decode_inference.ipynb  # Greedy decoding for all 4 models
├── report/
│   └── Deep_Learning_Midterm_Report.pdf
├── assets/
│   └── sample_result.jpg
├── requirements.txt
├── README.md
└── LICENSE
```

## How to Run
```bash
git clone https://github.com/<your-username>/vqa-cnn-lstm-attention.git
cd vqa-cnn-lstm-attention
pip install -r requirements.txt
# Open notebooks/Code_Midterm_DL.ipynb in Jupyter or Google Colab
```

## Future Work
- Replace LSTM decoder with a Transformer-based decoder for better long-range dependency handling.
- Expand dataset domain beyond human activities.
- Deploy as an interactive web demo (Gradio/Streamlit).

## Authors
- Huynh Nhat Tu
- Tran Duy Bao

Course project — Deep Learning, Ton Duc Thang University (2025), instructed by Dr. Le Anh Cuong.

## License
This project is licensed under the MIT License.
