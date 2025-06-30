import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
import json

# Paths
dataset_dir = r'C:\Users\Tamad\.cache\butterfly'
train_img_dir = os.path.join(dataset_dir, 'train')
test_img_dir = os.path.join(dataset_dir, 'test')
train_csv = os.path.join(dataset_dir, 'Training_set.csv')
test_csv = os.path.join(dataset_dir, 'Testing_set.csv')

# Load CSVs
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Add full paths to images
train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(train_img_dir, x))
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_img_dir, x))

# Add 'label' column to test set if needed (dummy for now)
test_df['label'] = 'unknown'

# Image data generators
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col=None,
    target_size=(224,   224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Load model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)


# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Starting training...")
model.fit(train_generator, validation_data=val_generator, epochs=20)
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
# Predict on test
print("🔍 Predicting on test set...")
predictions = model.predict(test_generator)

# Optional: Save predictions
predicted_classes = tf.argmax(predictions, axis=1).numpy()
test_df['predicted_label'] = predicted_classes
test_df.to_csv("test_predictions.csv", index=False)
print("✅ Predictions saved to test_predictions.csv")


# Save model
model.save("vgg16_butterfly_model.h5")
print("✅ Model saved as vgg16_butterfly_model.h5")
print(train_df['label'].value_counts())