import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

DATA_PATH = os.path.join('MP_Data')

actions = np.array([
    'Yes',
    'Please',
    'Thanks'
])

sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading training data...")

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_path):
        print(f"Folder missing: {action_path}")
        continue

    for sequence in os.listdir(action_path):
        sequence_path = os.path.join(action_path, sequence)

        if not os.path.isdir(sequence_path):
            continue

        window = []
        complete = True

        for frame_num in range(sequence_length):
            path = os.path.join(sequence_path, f"{frame_num}.npy")

            if not os.path.exists(path):
                complete = False
                break

            res = np.load(path)
            window.append(res)

        if complete:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Skipped incomplete folder: {sequence_path}")

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)

print("Total complete sequences:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    stratify=y,
    random_state=42
)

print(f"Data loaded! Train: {len(X_train)} | Test: {len(X_test)}")

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 1662)),
    BatchNormalization(),
    Dropout(0.2),

    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    BatchNormalization(),
    Dropout(0.2),

    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    BatchNormalization(),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(actions.shape[0], activation='softmax')
])

optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

log_dir = os.path.join('Logs')

callbacks = [
    TensorBoard(log_dir=log_dir),
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        'action_best.h5',
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nStarting Bidirectional LSTM training...")

history = model.fit(
    X_train,
    y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    batch_size=16
)

print("\n--- Final Evaluation ---")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

model.save('action.h5')

print("\nModel saved as 'action.h5'")
print("Best model also saved as 'action_best.h5'")