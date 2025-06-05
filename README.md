# BirdCLEF 2025 : Kaggle Competition Solution

Problem Statement - 

| Task                 | Library                                                      |
| -------------------- | ------------------------------------------------------------ |
| Audio loading & spec | `librosa`, `torchaudio`                                      |
| ML                   | `PyTorch`, `fastai`, `Keras`                                 |
| Pretrained models    | `torchvision`, `panns_inference`, `HuggingFace Transformers` |
| Visualization        | `matplotlib`, `seaborn`, `IPython.display.Audio`             |
| Audio augmentation   | `torch-audiomentations`, `audiomentations`                   |


---

## 📌 Project Overview

This repository contains my solution for the **\birdclef-2025** on [Kaggle](https://www.kaggle.com/competitions/birdclef-2025/overview). 

The main objective of the competition is to build machine learning models that can identify wildlife species from audio recordings collected in a Colombian rainforest reserve called El Silencio. These species can include birds, mammals, amphibians, insects, and more - any creature that makes a sound.

**Why is this important?**
* Monitoring biodiversity helps track the success or failure of forest restoration efforts.
* Traditional wildlife surveys (using humans) are expensive and slow.
* Passive Acoustic Monitoring (PAM) + ML can cover more ground faster and more frequently.

My final model achieved a **public leaderboard score of `0.663`**, and this README outlines the steps taken from data exploration to final model submission.

---

## 📌 Project Structure

The entire project is organized into the following key stages:

---

## 1. Understanding the Task & Data

**Goal**: Given 1-minute `.ogg` soundscape recordings, predict the probability of presence for each of the 206 bird species at every 5-second interval.

**Key Data Files**:

* `train_audio/`: Individual bird call recordings with species labels.
* `train_metadata.csv`: Metadata including `primary_label`, `filename`, location, and time.
* `test_soundscapes/`: Unlabeled soundscape recordings for inference.
* `sample_submission.csv`: Shows required format — one row per 5-second segment of test audio.
* `taxonomy.csv`: Provides label → scientific/common name mappings.

---

## 2. Exploratory Data Analysis (EDA)

I explored:

* **Label Distribution**: Identified class imbalance in bird species.
* **Location/Geography**: Verified metadata includes lat/lon and site info.
* **Time-of-day Effects**: Revied `date`, `time`, and their potential use.
* **Audio Characteristics**:
  * Duration, sampling rates, presence of noise
  * Identified long-tailed distribution of clips per species

EDA helped guide preprocessing, augmentation strategies, and model design.

---

## 3. Data Preprocessing

* **Audio Resampling**: Converted all audio to a consistent sample rate (32kHz).
* **Mono Conversion**: Averaged multi-channel audio to mono.
* **Segmentation**:
  * Training: Many-to-one classification on clips
  * Inference: Sliced test soundscapes into **non-overlapping 5-second segments**
* **Feature Extraction**:
  * Used **log-Mel Spectrograms** as input features (`n_mels=128`)
  * Normalized per-segment with z-score standardization

---

## 🧠 4. Model Architecture

Used a custom **BirdCLEFModel**, a CNN-based architecture designed to work on Mel Spectrograms. Key components:
* **Backbone**: CNN (e.g., ResNet-based or EfficientNet pretrained)
* **Classifier Head**: Fully connected layers projecting to 206 sigmoid outputs
* **Loss**: `BCEWithLogitsLoss()` to handle multi-label classification

---

## 📦 5. Model Loading (Fixing State Dict Mismatch)

Loaded a pretrained checkpoint from previous training. Since the model originally expected grayscale spectrograms (1 channel) and the current backbone expected RGB (3 channels):

* **Fixed size mismatch** in the first `conv1` layer using:

```python
new_ight = old_ight.repeat(1, 3, 1, 1) / 3.0
```

* Loaded modified state\_dict using `.load_state_dict(converted_state_dict)` after patching.

---

## 6. Inference Pipeline

For each test `.ogg` soundscape file:

1. **Load waveform**
2. **Preprocess**:

   * Resample + Mono
   * Slice into 5s segments (or pad if too short)
   * Convert to log-Mel spectrogram
3. **Model Prediction**:

   * Use `model(logmel).sigmoid()` to obtain 206 probabilities
4. **Build Submission**:

   * Match format of `sample_submission.csv`
   * Output: one row per 5s segment with `row_id` and 206 predicted probabilities

Fixed bug where submission file was empty due to `num_segments == 0` for short clips.

Final fix: ensured **at least one segment** is predicted per file:

```python
num_segments = max(1, total_samples // NUM_SAMPLES)
```

---

## 7. Submission Format

* Column 0: `row_id` = `soundscape_id_time`
* Columns 1–206: Predicted probabilities for each `species_id` from the sample submission.

Ensured the species columns in the submission match the `sample_submission.csv` column order exactly.

---

## Highlights

* Handled class imbalance via label-aware architecture
* Processed all audio in a consistent format using `torchaudio` and `torchvision.transforms`
* Reused and adapted a pretrained model for inference
* Fixed critical bugs in loading ights and generating submission format
* Efficiently segmented and predicted long test recordings

---
