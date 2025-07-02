# Hate Speech Detection for Amharic Language

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![HuggingFace Model](https://img.shields.io/badge/Model-EthioLLM--l--70K-purple)](https://huggingface.co/EthioNLP/EthioLLM-l-70K)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-brightgreen?logo=google-colab)](https://colab.research.google.com/github/shama-llama/hate-speech-detection/blob/main/src/hate_speech_detection.ipynb)

## Project Overview
This project provides a pipeline for detecting hate speech in Amharic social media and online text. It combines multiple Amharic hate speech datasets, and then applies preprocessing and normalization. The detection system uses a state-of-the-art transformer model and applies adversarial and hierarchical architectures for classification.

## Data Sources
The unified dataset is constructed from the following public Amharic hate speech datasets, each in its own subfolder under `dataset/`:

- **MD2023** ([Degu, 2023](https://doi.org/10.17632/k4xk27zcpr.3)): Amharic text dataset extracted from memes for hate speech detection or classification, focusing on Facebook image posts.
- **RANLP2023** ([Ayele et al., 2023](https://ranlp.org/2023)): Amharic hate speech data from the 14th International Conference on Recent Advances in Natural Language Processing (RANLP 2023), with standard train/dev/test splits.
- **SG2020** ([Getachew, 2020](https://doi.org/10.17632/ymtmxx385m.1)): Amharic Facebook dataset for hate speech detection, collected from social media posts and labeled for hate speech.
- **SM2022** ([Minale, 2022](https://doi.org/10.17632/p74pfhz3yx.1)): Amharic social media dataset for hate speech detection and classification, with deep learning-ready normalized data.
- **TRAC-HI2024** ([Ayele et al., 2024](https://aclanthology.org/2024.trac-1.1/)): TRAC 2024 Hindi/Amharic hate speech data, including category splits, from the Fourth Workshop on Threat, Aggression & Cyberbullying.
- **TRAC-HM2024** ([Ayele et al., 2024](https://aclanthology.org/2024.trac-1.1/)): TRAC 2024 Hindi/Amharic hate speech, preprocessed for modeling.
- **ZAK2021** ([Zeleke, 2021](https://doi.org/10.5281/zenodo.5036437)): Amharic Hate Speech Detection Dataset, published on Zenodo.

For citation, please use the following references:

- Ayele et al., 2024. Exploring Boundaries and Intensities in Offensive and Hate Speech: Unveiling the Complex Spectrum of Social Media Discourse. Proceedings of The Fourth Workshop on Threat, Aggression & Cyberbullying, Torino, Italy.
- Ayele et al., 2023. Exploring Amharic Hate Speech Data Collection and Classification Approaches. Proceedings of the 14th International Conference on RECENT ADVANCES IN NATURAL LANGUAGE PROCESSING (RANLP 2023), Varna, Bulgaria, pp. 49--59.
- Minale, Samuel (2022). Amharic Social Media Dataset for Hate Speech Detection and Classification in Amharic Text with Deep Learning. Mendeley Data, V1. doi:10.17632/p74pfhz3yx.1
- Getachew, Surafel (2020). Amharic Facebook Dataset for Hate Speech detection. Mendeley Data, V1. doi:10.17632/ymtmxx385m.1
- Degu, Mequanent (2023). Amharic text dataset extracted from memes for hate speech detection or classification. Mendeley Data, V3. doi:10.17632/k4xk27zcpr.3
- Zeleke Abebaw Kassa (2021). Amharic Hate Speech Detection Dataset. Zenodo, Jun. 28, 2021. doi:10.5281/zenodo.5036437.

All these are merged into `dataset/combined_dataset.csv` and further processed into `dataset/preprocessed_dataset.csv` for modeling.

## Features
- **Dataset Integration & Preprocessing**
  - Merges multiple Amharic hate speech datasets
  - Cleans, normalizes, and maps labels for consistency
  - Splits data into train/dev/test for evaluation
- **Exploratory Data Analysis (EDA)**
  - Visualizes label, source, and text length distributions
  - Detects duplicates and missing values
  - N-gram (unigram, bigram, trigram) analysis per class
  - Word cloud visualizations for each class
  - Punctuation and special character analysis
- **Adversarial Data Augmentation**
  - Generates new hate speech examples using masked language modeling to improve model robustness
- **Model Training & Evaluation**
  - Fine-tunes [EthioLLM-l-70K](https://huggingface.co/EthioNLP/EthioLLM-l-70K) and other transformer models
  - Supports adversarial and hierarchical architectures
  - Reports accuracy, F1, precision, and recall on held-out test data
- **Visualization**
  - All plots and word clouds are saved to `output/images` for easy access and publication

## Directory Structure
```
dataset/
    combined_dataset.csv
    preprocessed_dataset.csv
    [subdirectories contatining the raw datasets]
output/
    images/             # All generated plots and word clouds
    ahta_model_results/ # Model output
src/
    dataset_analysis.ipynb
    dataset_preprocessing.ipynb
    hate_speech_detection.ipynb
```

## Notebooks
- **dataset_analysis.ipynb**: In-depth EDA, n-gram, word cloud, and punctuation analysis of the combined dataset.
- **dataset_preprocessing.ipynb**: Data cleaning, normalization, label mapping, and train/dev/test split creation for all source datasets.
- **hate_speech_detection.ipynb**: End-to-end pipeline for adversarial data augmentation, model training, and evaluation using transformer models.

## How to Run
1. **Install Requirements**
   - Python 3.8+
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - For word cloud and Amharic text support, install:
     ```bash
     pip install wordcloud matplotlib seaborn scikit-learn pandas tqdm transformers accelerate
     sudo dnf install fonts-noto fonts-noto-cjk fonts-noto-color-emoji fonts-noto-unhinted fonts-noto-extra fonts-noto-mono fonts-noto-ui-core fonts-noto-ui-extra fonts-noto-vf fonts-abyssinica
     ```
2. **Prepare Data**
   - Place all raw and processed datasets in the `dataset/` directory as shown above.
   - If necessary, run the `dataset_builder.py` script to regenerate the combined dataset.
3. **Run Notebooks**
   - Open and execute the notebooks in order for EDA, preprocessing, and model training.
   - All generated images will be saved in `output/images`.

## Notes
- For best results with Amharic text, use an Ethiopic font (e.g., Noto Sans Ethiopic or Abyssinica SIL) installed and configured in matplotlib.
- The adversarial augmentation step is optional but recommended for improving model performance and robustness.
- The code is designed to run both locally and in Google Colab (see the Colab badge in the main notebook).

## License
This project is licensed under the [MIT License](LICENSE).
