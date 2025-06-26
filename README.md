# Curriculum Contrastive EEG-to-Text Model

A deep learning framework that aligns EEG brain signals with text using curriculum contrastive learning.

## Overview

This project implements a novel approach to map brain signals recorded during natural reading to their corresponding text representations. The model uses dual encoder networks that learn a shared semantic space where EEG and word embeddings are aligned through contrastive learning.

**Key Innovation**: Curriculum learning strategy that progressively trains the model from simple to complex examples, mimicking human learning behavior.

## Features

- **Dual Encoder Architecture** - Separate encoders for EEG signals and text
- **Curriculum Learning** - Progressive training from simple to complex examples  
- **Contrastive Learning** - InfoNCE loss for bidirectional EEG-text alignment
- **Transformer-based Processing** - Advanced EEG signal processing with attention
- **Two-Step Training** - Optimized training pipeline with adaptive learning rates

## Quick Start

### Installation

```bash
git clone https://github.com/amar3012005/Contrastive_EEG.git
cd Contrastive_EEG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python scripts/train.py --config config/config.yaml

# Custom parameters
python scripts/train.py \
    --config config/config.yaml \
    --batch-size 16 \
    --step1-epochs 30 \
    --step2-epochs 15
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model-path output/best_model.pt \
    --data-path data/zuco_dataset.pickle
```

## Dataset

**Source**: ZuCo EEG2Text Dataset (Subject ZJN)

- **Sessions**: 8 reading sessions (SR1-SR8)
- **Channels**: 105 EEG channels
- **Words**: 772 word-level recordings
- **Signal Length**: 65-2,314 timepoints per word
- **Format**: NPZ files for EEG data, CSV for metadata

## Architecture

### EEG Encoder
```
EEG Input (105 channels) → Linear Projection → Positional Encoding → 
Transformer Layers → Global Pooling → Projection Head → 128D Embedding
```

### Text Encoder
```
Text Tokens → BART Encoder → CLS Token → 
Linear Projection → Layer Norm → 128D Embedding
```

### Contrastive Learning
- **Loss Function**: InfoNCE (bidirectional)
- **Temperature**: 0.07
- **Similarity**: Cosine similarity between embeddings
- **Objective**: Maximize mutual information between EEG and text

## Training Strategy

### Curriculum Learning Phases

1. **Basic**: High-frequency words, simple EEG patterns
2. **Intermediate**: Medium complexity examples
3. **Advanced**: Complex words, high-variance EEG signals

### Two-Step Process

**Step 1 - Foundation Training**
- Frozen BART layers (except embeddings)
- Higher learning rate for EEG encoder
- Focus on basic alignment

**Step 2 - Fine-tuning**
- All parameters trainable
- Lower learning rate
- Refined semantic alignment

## Results

The model produces:

- **Training Metrics** - Loss curves and accuracy over time
- **Embedding Visualizations** - t-SNE plots of learned representations
- **Similarity Matrices** - EEG-text alignment patterns
- **Cross-modal Retrieval** - Precision@K and recall metrics

## Evaluation Metrics

- **Contrastive Accuracy** - Correct matches in similarity ranking
- **InfoNCE Loss** - Contrastive learning objective
- **Cross-modal Retrieval** - Precision@K, Recall@K, MRR
- **Embedding Quality** - t-SNE visualization analysis

## Applications

- Brain-computer interfaces
- Neurolinguistic research
- Cognitive modeling
- Medical diagnostics
- Human-AI interaction

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers (Hugging Face)
- NumPy, Scikit-learn
- Matplotlib, Seaborn

## Troubleshooting

**NumPy Issues**
```bash
pip uninstall numpy
pip install numpy==1.24.3
```

**CUDA Memory**
- Reduce batch size
- Enable gradient checkpointing

**Dataset Path**
- Verify ZuCo dataset location
- Update paths in config file

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Open Pull Request

## References

- **ZuCo Dataset**: [Hollenstein et al., 2018](https://www.nature.com/articles/sdata2018291)
- **InfoNCE Loss**: [van den Oord et al., 2018](https://arxiv.org/abs/1807.03748)
- **BART**: [Lewis et al., 2019](https://arxiv.org/abs/1910.13461)
