# Image Classification using Convolutional Neural Networks (CNN)

**Author:** Rudhra Sitholey  
**Registration Number:** 23BCY10296  
**Assignment:** AI-ML Assignment – 9  
**GitHub Repository:** [Image-Classification-using-Convolutional-Neural-Networks--CNN-](https://github.com/rudhra23bcy10296/Image-Classification-using-Convolutional-Neural-Networks--CNN-)

---

## 📌 Objective

An animal welfare organization wants to automate the classification of pet images into **Cats** and **Dogs**. The objective of this project is to develop, train, and evaluate a **Convolutional Neural Network (CNN)** model to accurately classify pet images.

---

## 🔗 Dataset Link

- **Dataset Name:** Kaggle Dog and Cat Classification Dataset (`PetImages`)
- **Kaggle URL:** [Dog and Cat Classification Dataset on Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Dataset Structure:**
  - `PetImages/Cat`: 12,499 images
  - `PetImages/Dog`: 12,499 images
  - **Total Images:** 24,998 images
  - **Classes:** 2 (`Cat`, `Dog`)

---

## 🛠️ Libraries Used

- **Deep Learning Framework:** `TensorFlow`, `Keras` (`Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `ImageDataGenerator`)
- **Data Manipulation & Processing:** `NumPy`, `Pillow (PIL)`
- **Model Evaluation:** `scikit-learn` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`)
- **Visualization:** `Matplotlib`, `Seaborn`

---

## 🔬 Methodology

1. **Data Understanding & Inspection:**
   - Inspected folder structure (`PetImages/Cat` and `PetImages/Dog`).
   - Visualized 5 sample images alongside their target class labels.
   - Identified 2 target classes (`Cat`, `Dog`), variable raw image dimensions, and a total of 24,998 image samples.

2. **Data Preprocessing:**
   - Resized all images to a uniform resolution of **128 × 128 pixels**.
   - Normalized pixel values from $[0, 255]$ to range $[0, 1]$ via float division ($1/255.0$).
   - Split dataset into an **80% Training set** (20,000 samples) and a **20% Testing set** (4,998 samples).
   - Created data generators using TensorFlow/Keras `ImageDataGenerator`.

3. **Model Development & Architecture:**
   - Built a 9-layer CNN using Keras `Sequential` API.
   - Compiled with **Adam** optimizer, **Binary Crossentropy** loss function, and **Accuracy** metric.
   - Trained model for **10 epochs**.

4. **Model Evaluation:**
   - Calculated test accuracy, precision, recall, and F1-score.
   - Generated confusion matrix heatmap.
   - Plotted **Accuracy vs. Epoch** and **Loss vs. Epoch** performance graphs.

---

## 🏗️ CNN Architecture

| Layer | Type | Output Shape | Filter / Neurons | Activation | Parameters |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 1** | Conv2D | `(None, 126, 126, 32)` | 32 filters (3×3) | ReLU | 896 |
| **Layer 2** | MaxPooling2D | `(None, 63, 63, 32)` | Pool Size (2×2) | - | 0 |
| **Layer 3** | Conv2D | `(None, 61, 61, 64)` | 64 filters (3×3) | ReLU | 18,496 |
| **Layer 4** | MaxPooling2D | `(None, 30, 30, 64)` | Pool Size (2×2) | - | 0 |
| **Layer 5** | Conv2D | `(None, 28, 28, 128)` | 128 filters (3×3) | ReLU | 73,856 |
| **Layer 6** | MaxPooling2D | `(None, 14, 14, 128)` | Pool Size (2×2) | - | 0 |
| **Layer 7** | Flatten | `(None, 25088)` | - | - | 0 |
| **Layer 8** | Dense | `(None, 128)` | 128 Neurons | ReLU | 3,211,392 |
| **Layer 9** | Output (Dense) | `(None, 1)` | 1 Neuron | Sigmoid | 129 |

- **Total Trainable Parameters:** 3,304,769

---

## 📊 Results & Performance

- **Test Accuracy:** **80.47%**
- **Precision:** **0.81**
- **Recall:** **0.80**
- **F1-Score:** **0.80**

### Performance Observations

1. **Steady Metric Convergence:** Training accuracy steadily increased over 10 epochs while binary crossentropy loss consistently decreased.
2. **Balanced Classification:** High, balanced precision (0.81) and recall (0.80) across both Cat and Dog classes.
3. **Generalization Capabilities:** Validation accuracy closely tracked training accuracy without severe overfitting.
4. **Impact of Convolutional Hierarchy:** Successive convolution and max-pooling layers effectively extracted spatial features and abstract visual patterns.

---

## 🎯 Conclusion

### Summary & Key Findings

This project successfully developed and evaluated a 9-layer **Convolutional Neural Network (CNN)** for binary image classification of Cats vs. Dogs. By standardizing input images to $128 \times 128$ pixels and normalizing pixel intensities to $[0, 1]$, the model achieved **80.47% accuracy** on unseen test data.

**Importance of Convolution and Pooling Layers:** Convolutional layers apply spatial filter kernels to automatically detect local features (edges, textures, shapes), while max-pooling layers downsample feature maps to reduce spatial dimensions, lower computational complexity, and enforce translation invariance.

**Advantage of CNN over ANN for Image Classification:** Unlike traditional Artificial Neural Networks (ANNs) which require flattening 2D/3D images into 1D vectors thereby destroying spatial topology and incurring millions of redundant trainable parameters, CNNs preserve 2D grid structure and utilize parameter sharing across channels.

**Limitation of CNN:** CNNs require substantial amounts of labeled training data and high computational memory (GPU acceleration) to avoid overfitting and achieve high accuracy on complex, high-resolution real-world visual datasets.
