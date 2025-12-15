# Bone Age Prediction from Hand Radiographs

A deep learning–based system for automated pediatric bone age assessment from hand X-ray images.  
This project addresses both **continuous age prediction (regression)** and **developmental stage classification**, with a strong focus on **accuracy, interpretability, and fairness**.

---

## 📌 Problem Overview

Bone age assessment is a critical diagnostic tool in pediatrics to evaluate skeletal maturity and detect growth-related disorders.  
Traditional methods such as **Greulich–Pyle** and **Tanner–Whitehouse** are manual, time-consuming, and subject to inter-observer variability.

This project aims to:
- Predict **continuous bone age** from hand X-ray images
- Classify patients into **developmental age stages**
- Provide **interpretable predictions**
- Ensure **fair performance across genders**

---

## 📊 Dataset

- **Dataset:** RSNA Pediatric Bone Age Dataset
- **Images:** 12,611 pediatric hand radiographs
- **Age Range:** 1–228 months (0–19 years)
- **Gender Distribution:** ~54% Male, ~46% Female
- **Data Split:**
  - Training: 70%
  - Validation: 15%
  - Holdout Test: 15% (stratified by age and gender)

---

## 🧠 Methodology

### Image Preprocessing
- Resized all images to **384×384**
- Normalized using **ImageNet statistics**
- Converted grayscale X-rays to **3-channel RGB**
- Applied medically safe data augmentation:
  - Rotation (±15°)
  - Zoom, brightness, contrast
  - Horizontal flipping (no vertical flips)

---

## 🏗 Model Architecture

### Regression Model
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Sex-aware embedding:** 32-dimensional biological sex embedding
- **Loss Function:** Huber Loss
- **Output:** Continuous bone age (years)

### Classification Model
- Same backbone and embeddings
- **Output:** 4 age categories
  - Infant/Toddler (0–5 years)
  - Child (5–10 years)
  - Pre-adolescent (10–15 years)
  - Adolescent (15–19 years)

### Ensemble Learning
- Extracted deep features from CNN
- Trained an **XGBoost regressor**
- Final prediction:  
  **0.7 × CNN + 0.3 × XGBoost**

---

## ⚙️ Training Details

- **Optimizer:** AdamW
- **Learning Rate:** Cosine Annealing with warmup
- **Batch Size:** 16
- **Early Stopping:** Patience = 10
- **Regularization:** Dropout + Gradient Clipping
- **Training Strategy:** Progressive fine-tuning

---

## 📈 Results

### Regression Performance (Holdout Set)
- **MAE:** 7.38 months
- **RMSE:** 9.75 months
- **R² Score:** 0.9431

### Classification Performance
- **Accuracy:** 88.9%
- **Quadratic Weighted Kappa (QWK):** 0.895

These results fall within the accuracy range of **clinical experts**.

---

## 🔍 Interpretability & Analysis

- **Grad-CAM** visualizations show attention on clinically relevant regions:
  - Carpal bones
  - Epiphyseal plates
  - Metacarpals and phalanges
- **t-SNE visualizations** confirm meaningful age-ordered feature representations
- **Error analysis** shows near-Gaussian error distribution with minimal outliers

---

## ⚖️ Fairness & Bias Evaluation

- Gender-wise MAE difference: **~1.3 months**
- No systematic over- or under-estimation across genders
- Sex-aware embeddings help capture biological differences without bias

---

## 🏥 Clinical Relevance

Potential applications:
- Pediatric screening tool
- Decision support for radiologists
- Large-scale growth studies
- Telemedicine in resource-limited settings

⚠️ *This model is for research purposes only and is not clinically certified.*

---

## 🔮 Future Work

- Uncertainty-aware predictions
- Transformer-based attention mechanisms
- Multi-center dataset validation
- Self-supervised and federated learning
- Deployment-ready inference pipelines

---

## 🛠 Tech Stack

- **Languages:** Python
- **Deep Learning:** PyTorch / TensorFlow
- **ML:** scikit-learn, XGBoost
- **Visualization:** Matplotlib, Grad-CAM, t-SNE

---

## 📚 References

- RSNA Pediatric Bone Age Challenge (2017)  
  https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017

---

## ⭐ Key Takeaway

This project demonstrates that **efficient deep learning models** can achieve **clinically relevant accuracy**, provide **interpretable predictions**, and maintain **fairness across demographic groups** in pediatric bone age assessment.
