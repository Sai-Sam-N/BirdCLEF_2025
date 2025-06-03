## 🧠 Key Understanding of Dataset

### 🟢 **Training Data**

* **train.csv** → Labels & metadata for training.
* **train\_audio/** → Short labeled clips (bird/amphibian/mammal/insect calls).
* Label = `primary_label` (206 possible species)

### 🔵 **Unlabeled Audio**

* **train\_soundscapes/** → Long-form, unlabeled 1-minute recordings (optional for semi-supervised learning)

### 🔴 **Test Data**

* **test\_soundscapes/** → \~700 1-min clips
* Must predict **which species are present**, **every 5s** (so: 12 segments per clip)

---

## 🛠️ Final Deliverable

A notebook that:

* Loads each test soundscape
* Splits it into 5-second chunks
* Classifies all 206 species (multi-label)
* Outputs a `sample_submission.csv` with probability scores

---

## 🧪 8-Hour Execution Plan

### ⏱️ HOUR 0–1: Dataset Setup + EDA

* ✅ Load `train.csv`
* ✅ Explore label distribution (head/tail species)
* ✅ Visualize class imbalance
* ✅ Load & plot a few `.ogg` files from `train_audio/` using `librosa`
* ✅ Optional: Map a few `primary_label`s to taxonomy with `taxonomy.csv`

### ⏱️ HOUR 1–2.5: Feature Engineering

* ✅ Convert `.ogg` to **log-mel spectrograms**

  * Use `librosa.feature.melspectrogram()` or `torchaudio.transforms`
* ✅ Normalize/pad spectrograms to fixed length (e.g., 5s)
* ✅ Save them as tensors/images (to avoid repeated reprocessing)
* ✅ Define 5s windowing logic for test soundscapes

### ⏱️ HOUR 2.5–4.5: Baseline Model Training

* ✅ Model: Pretrained CNN (e.g., `ResNet18`, `EfficientNetB0`) via `torchvision.models` or `Keras`
* ✅ Input: `(1, H, W)` grayscale mel-spec
* ✅ Output: 206 sigmoid activations (multi-label binary classification)
* ✅ Loss: `BCEWithLogitsLoss`
* ✅ Eval: `Average Precision`, `Label-weighted F1`, etc.

### ⏱️ HOUR 4.5–6: Inference Pipeline + Submission

* ✅ Load each `.ogg` test soundscape
* ✅ Chunk into 12 × 5s segments
* ✅ Apply model to each chunk
* ✅ Output predictions: `[row_id] + [206 probabilities]`
* ✅ Format as per `sample_submission.csv`

### ⏱️ HOUR 6–8: Semi-Supervised Boost (Optional but Valuable)

#### Option A: Pseudo-Labeling

* Train model on labeled `train_audio`
* Predict on `train_soundscapes` with confidence threshold (e.g., > 0.9)
* Retrain with augmented data

#### Option B: Pretrained Audio Embeddings

* Use `PANNs` (Pretrained Audio Neural Networks) or `Wav2Vec2`
* Extract 1024-D features → Train classifier head on top

---

## 📚 Recommended Libraries

| Task                 | Library                                                      |
| -------------------- | ------------------------------------------------------------ |
| Audio loading & spec | `librosa`, `torchaudio`                                      |
| ML                   | `PyTorch`, `fastai`, `Keras`                                 |
| Pretrained models    | `torchvision`, `panns_inference`, `HuggingFace Transformers` |
| Visualization        | `matplotlib`, `seaborn`, `IPython.display.Audio`             |
| Audio augmentation   | `torch-audiomentations`, `audiomentations`                   |

---
