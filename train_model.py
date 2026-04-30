import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ============================================================
# SIGNBRIDGE - BIDIRECTIONAL LSTM MODEL TRAINING
# Run this SECOND after data_collection.py
# ============================================================

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['Hello', 'Thanks', 'ILoveYou'])
no_sequences = 60
sequence_length = 30

# --- LOAD DATA ---
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading training data...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
print(f"Data loaded! Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# BIDIRECTIONAL LSTM MODEL
# Key Improvement: Bidirectional layers read the sequence
# FORWARD and BACKWARD, capturing richer gesture patterns.
# ============================================================
model = Sequential([
    # Layer 1: Bidirectional LSTM - learns patterns in both directions
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 1662)),
    BatchNormalization(),
    Dropout(0.2),

    # Layer 2: Bidirectional LSTM
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    BatchNormalization(),
    Dropout(0.2),

    # Layer 3: Bidirectional LSTM (no return_sequences for Dense next)
    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    BatchNormalization(),
    Dropout(0.2),

    # Dense layers for classification
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    # Output layer
    Dense(actions.shape[0], activation='softmax')
])

# --- COMPILE ---
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# --- CALLBACKS ---
log_dir = os.path.join('Logs')
callbacks = [
    TensorBoard(log_dir=log_dir),
    # Stop early if validation doesn't improve for 30 epochs
    EarlyStopping(monitor='val_categorical_accuracy', patience=30, restore_best_weights=True, verbose=1),
    # Reduce LR if stuck
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6),
    # Save the best model automatically
    ModelCheckpoint('action_best.h5', monitor='val_categorical_accuracy', save_best_only=True, verbose=1)
]

# --- TRAIN ---
print("\nStarting Bidirectional LSTM training...")
history = model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    batch_size=16
)

# --- EVALUATE ---
print("\n--- Final Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- SAVE ---
model.save('action.h5')
print("\nModel saved as 'action.h5'")
print("Best model also saved as 'action_best.h5'")
print("\nTip: If accuracy < 90%, use 'action_best.h5' - it's the checkpoint of the best epoch.")