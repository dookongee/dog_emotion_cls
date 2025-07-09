# 🐶 Dog Emotion Classification

This project aims to detect dog emotions such as **happy** or **not happy** based on facial expressions. It covers the full pipeline from data collection to ensemble modeling, with a focus on real-world challenges like breed imbalance and emotional ambiguity.

---

## 📁 Project Overview

- **Goal**: Classify a dog's emotion using only image input
- **Scope**:
  - Collected and refined images from Unsplash and Stanford Dog Dataset
  - Built multiple CNN models (ResNet, DenseNet, EfficientNetB0)
  - Applied ensemble methods (soft voting)
  - Developed pre-filtering to detect whether a dog is present in the image

---

## 📊 Dataset & Labeling

- 1st stage: Images from Unsplash (`happy`, `angry`)
- 2nd stage: Stanford Dogs Dataset with manual labeling
- Used cosine similarity + representative images for data filtering
- Implemented a **semi-automatic labeling strategy** using a pretrained model

---

## 🧠 Models & Training

- Backbones: ResNet18, ResNet34, DenseNet, EfficientNetB0
- Optimizers: SGD, AdamW
- Schedulers: StepLR, CosineAnnealing, ReduceLROnPlateau
- Ensemble: Top 7 models using **soft voting**

---

## 🐾 Dog Detection Step

To prevent unreliable emotion predictions on images without dogs:

- Used a pretrained **ResNet-34** to detect whether a dog is present  
- If a dog is **not detected**, a warning message is shown  
- Emotion label & probability are still output (in case of misdetection), but marked as **less reliable**

---

## 🔮 Future Work

- Extend to **multi-label emotion classification**: happy, hungry, sleepy, etc.  
- Use `BCEWithLogitsLoss` for overlapping emotional states  
- Improve breed balance and data diversity

---

## 📷 Sample Output

![스크린샷 2025-07-09 163558](https://github.com/user-attachments/assets/bc897e93-e3bb-4237-8115-1037c2ab106c)
![스크린샷 2025-07-09 163628](https://github.com/user-attachments/assets/52fde026-14c4-4791-936a-4cccf62a1a26)
![스크린샷 2025-07-09 163729](https://github.com/user-attachments/assets/96e79b6a-78e9-4f87-b592-a910ef319435)


---

## 📎 Reference

This project was created as part of my preparation for graduate study in computer vision.  
[Full project documentation on Notion →](https://invincible-gargoyle-054.notion.site/Dog-Emotion-Classification-229c4ba53ecb806087d3fc7afe7de787?pvs=73)
[Live Demo →](https://dogemotioncls-333ewtsbckryqqfngf6ksm.streamlit.app/)
