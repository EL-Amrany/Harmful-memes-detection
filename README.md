# GuardHarMem: A Multimodal Framework for Harmful Meme Classification

## Overview
GuardHarMem is a multimodal framework and dataset for fine-grained harmful meme classification. It integrates text, image, and caption features for robust detection of harmful memes categorized as Racism, Mockery, and Promotion of Harmful Substances. This project addresses limitations in existing datasets and frameworks by providing fine-grained annotations and leveraging multimodal learning.

## Features
- **Dataset**: GuardHarMem includes 15,300 training samples with fine-grained labels for harmful content. It incorporates captions generated by GPT-4 and diverse data augmentation techniques.
  Images can be found here : https://zenodo.org/records/15388028
- **Multimodal Framework**: Combines features from text (BERT), images (VGG19), and captions to improve classification accuracy.
- **Performance**: Achieved F1-scores of 0.91 for binary classification and 0.82 for multiclass classification, outperforming baseline methods.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/GuardHarMem.git
   cd GuardHarMem

## Citation
@article{GuardHarMem2024,
  author = {Samir El-amrany, Salima Lamsiyah, Matthias R. Brust, Pascal Bouvry},
  title = {GuardHarMem: A Multimodal Dataset and Framework for Fine-Grained Harmful Meme Classification},
  journal = {To be added},
  year = {2024}
}
