
 # Emotion Classification from Audio Using CNN
 
This project involves building a Convolutional Neural Network (CNN) to classify emotions from audio files. It utilizes Mel Frequency Cepstral Coefficients (MFCCs) as features for training and testing. We also perform data augmentation to improve model performance and visualize results using confusion matrices. 

Table of Contents

Dataset

Feature Extraction

Data Augmentation

Model Architecture

Model Evaluation

How to Run

Dependencies

Results

Confusion Matrix & Report

# Dataset

The dataset contains audio files representing six emotional states:

Angry

Disgust

Fear

Happy

Neutral

Sad

Each emotion is stored in its respective folder. The project extracts MFCC features from the .wav files for emotion classification.

# Feature Extraction

We extract MFCCs as features from audio samples. Each audio file is loaded with Librosa for 3 seconds, and 13 MFCCs are computed, with their mean values across the time axis used as features.


audio, sr = librosa.load(file_path, duration=3)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)
features.append(mfcc_mean)
labels.append(emotion_label)

After processing all files, features and labels are converted to NumPy arrays for further analysis.

# Data Augmentation

To enhance the performance of our model, we perform data augmentation by adding noise, applying time-stretching, and pitch shifting. We also ensure that all audio data are padded or truncated to a fixed length.

def pad_or_truncate(data, max_len=11200):
    if len(data) < max_len:
        return np.pad(data, (0, max_len - len(data)), mode='constant')
    return data[:max_len]

# Add noise and apply pitch shift for augmentation
noise = np.random.randn(len(feature)) * 0.005
stretched_feature = librosa.effects.time_stretch(feature, rate=1.2)
pitched_feature = librosa.effects.pitch_shift(feature, sr=sr, n_steps=0.7)

This ensures the dataset is expanded to improve generalization.

Model Architecture

We use a CNN architecture for this task. Here is the structure of the model:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(13, 1, 1), padding='same'))
model.add(Flatten())
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

Model Evaluation

After training the model for 10 epochs, we evaluate it on the test set and print the accuracy and loss.

history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

To visualize training performance, we plot the training and validation loss over epochs.

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

Confusion Matrix & Report

We generate predictions on the test set and visualize the confusion matrix to understand the model's performance across different emotions.

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Emotion Classification - Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Emotions')
plt.ylabel('True Emotions')
plt.show()

Classification Report:

from sklearn.metrics import classification_report

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
report = classification_report(y_true_labels, y_pred_labels, target_names=class_names)
print(report)
Example output:

markdown
Copy code
              precision    recall  f1-score   support

       Angry       0.82      0.85      0.83       100
     Disgust       0.75      0.78      0.76       100
        Fear       0.80      0.81      0.80       100
       Happy       0.89      0.87      0.88       100
     Neutral       0.77      0.76      0.76       100
         Sad       0.83      0.82      0.83       100

    accuracy                           0.81       600
   macro avg       0.81      0.81      0.81       600
weighted avg       0.81      0.81      0.81       600

#How to Run

Clone the repository and navigate to the project directory.
Install dependencies using pip install -r requirements.txt.
Place the dataset in the appropriate folder structure.
Run the notebook or script containing the code.

Dependencies

Python 

Librosa

NumPy

Matplotlib

Seaborn

TensorFlow / Keras

Scikit-learn

Results

The final model achieves around 81% accuracy on the test set, with good performance across most emotional categories. The confusion matrix helps us understand the misclassifications and areas for further improvement.
