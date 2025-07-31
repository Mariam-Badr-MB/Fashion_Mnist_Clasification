# 🧠 Fashion MNIST Classification

A deep learning project to classify images of fashion items (e.g., shirts, trousers, shoes) using the Fashion-MNIST dataset and Convolutional Neural Networks (CNNs).

---

## 📁 Dataset

- **Source**: [Fashion-MNIST by Zalando](https://github.com/zalandoresearch/fashion-mnist)
- **Classes** (10):
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot
- **Data Format**:
  - 28x28 grayscale images
  - 60,000 training images
  - 10,000 test images

---
---

## 🧪 Steps Overview

### 1. 📦 Import Libraries
- `TensorFlow`, `Keras`, `NumPy`, `Matplotlib`, etc.

### 2. 📊 Data Loading and Preprocessing
- Load dataset using `tf.keras.datasets.fashion_mnist.load_data()`.
- Normalize images to range `[0, 1]`.
- Reshape images to include channel dimension: `(28, 28, 1)`.
- Shuffle and split validation set from training data.

### 3. 🧠 Model Architecture
Example CNN:
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])
```

### 4. ⚙️ Compile Model
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 5. 🏋️ Training
```python
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=256,
    callbacks=[early_stop_cb],
    shuffle=True
)
```

---

## 📈 Evaluation & Prediction

- Evaluate model on test set using `.evaluate()`.
- Predict with `.predict()` 
---

## ✅ Results
 - accuracy = 90%
---

## 📈 Project Workflow

1. **Load & Preprocess Data**
2. **Build CNN Model**
3. **Train the Model with Validation**
4. **Evaluate Accuracy and Loss**
5. **Visualize Results**

---

## 🔍 Example Output

- Training vs. Validation Accuracy  
- Training vs. Validation Loss  
- Model performance on test set  
- Sample predictions with true vs. predicted labels

---


## 👩‍💻 Author

**Mariam Badr**  
Faculty of Computers & Artificial Intelligence, Cairo University  
[GitHub](https://github.com/Mariam-Badr-MB) – [LinkedIn](https://www.linkedin.com/in/mariambadr13/)

---

## 📜 License

This project is for **educational and learning purposes only**.
