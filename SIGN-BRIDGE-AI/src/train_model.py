import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Dataset Path
INPUT_DIR = './processed_data'
MODEL_SAVE_PATH = './models/asl_model.h5'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10

# Optimized Dataset Loading with Augmentation
# Note: Augmentation is crucial for "perfecting" the model!
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True, # ASL gestures are hand-specific, but augmentation helps robustness
    validation_split=0.2
)

# Load training and validation datasets
train_generator = train_datagen.flow_from_directory(
    INPUT_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=123
)

val_generator = train_datagen.flow_from_directory(
    INPUT_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False,
    seed=123
)

class_names = list(train_generator.class_indices.keys())
print(f"Detected Classes: {class_names}")
if len(class_names) == 0:
    print("Error: No classes found in processed_data! Check directory structure.")
    exit(1)
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {val_generator.samples}")

# --- IMPROVED ARCHITECTURE: MobileNetV2 ---
# Using MobileNetV2 for faster inference and better performance in real-time.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))

# Fine-tuning: Freeze initial layers, unfreeze the top-level blocks
for layer in base_model.layers[:-15]:
    layer.trainable = False

# Add head layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Callbacks for better training
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=3, 
    factor=0.5, 
    min_lr=1e-7, 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

# --- TRAINING ---
print("Training specialized MobileNetV2 Model...")

if not os.path.exists('./models'):
    os.makedirs('./models')

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping]
)

# Final Save
model.save(MODEL_SAVE_PATH)
print(f"Success! Perfected model saved to: {MODEL_SAVE_PATH}")
