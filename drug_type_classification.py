import pandas as pd

df = pd.read_csv(r'path\active listings.csv')

columns_to_analyze = ['category1', 'category2', 'category3']
for column in columns_to_analyze:
    print(f"Analyzing: {column}")
    value_counts = df[column].value_counts(dropna=False)
    percentage = df[column].value_counts(dropna=False, normalize=True) * 100
    stats_df = pd.DataFrame({'Counts': value_counts, 'Percentage': percentage})
    print(stats_df)
    print("\n")

# Define drug catehpries
drug_categories = [
    "Stimulants", "Ecstasy", "Opioids", "Psychedelics", "Benzos", "Prescription", "Steroids", "Dissociatives",
    "Drugs", "Tobacco", "Weight Loss ", "Other"
]

def classify_category(row):
    if row == "Cannabis & Hashish":
        return 'cannabis'
    elif row in drug_categories:
        return 'non-cannabis'
    else:
        return 'non-drugs'

# Apply categories
df['new_category'] = df['category2'].apply(classify_category)
df.rename(columns={'blurb (description)': 'description'}, inplace=True)

df2 = pd.read_csv(r'path\drug_listings.csv')

#######################
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
texts = df['description'].tolist()
labels = df['new_category'].map({'cannabis': 0, 'non-cannabis': 1, 'non-drugs': 2}).tolist()

# Ensure all items in 'texts' are strings
texts = [str(item) if not isinstance(item, str) else item for item in texts]
# Now, check again to ensure all items are strings
print(all(isinstance(item, str) for item in texts))  # This should now print True

encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors="np")

from sklearn.model_selection import train_test_split
import tensorflow as tf

# Assume `labels` is your list of integer labels corresponding to 'texts'
input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']

# Split data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=2021, test_size=0.1)
train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, random_state=2021, test_size=0.1)

# Convert all of our data into tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_inputs, "attention_mask": train_masks},
    tf.one_hot(train_labels, depth=3)  # Assuming 3 classes: cannabis, non-cannabis, non-drugs
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": val_inputs, "attention_mask": val_masks},
    tf.one_hot(val_labels, depth=3)
))

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(100).batch(16)
val_dataset = val_dataset.batch(16)

from transformers import TFAutoModelForSequenceClassification

model_name = "emilyalsentzer/Bio_ClinicalBERT"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Compile the TensorFlow model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
# Train the model
# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=1,  # How many epochs to wait after the last improvement
    verbose=1,  # Print a message when stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric.
)

# Train the model with the EarlyStopping callback
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]  # Add the EarlyStopping callback here
)


# Evaluate the model
val_predictions = model.predict(val_dataset)
# Get predicted labels
predicted_labels = np.argmax(val_predictions.logits, axis=1)
true_labels = []
for batch in val_dataset.unbatch():
    true_labels.append(np.argmax(batch[1]))

true_labels = np.array(true_labels)
from sklearn.metrics import classification_report

# Generate Classification Report
report = classification_report(true_labels, predicted_labels, target_names=['cannabis', 'non-cannabis', 'non-drugs'])
print(report)

# Save Model
model.save(r'path\DNM_model', save_format='tf')


# Prediction on the other listing data
import numpy as np
descriptions = df2['product_description'].astype(str).tolist()
encodings = tokenizer(descriptions, truncation=True, padding="max_length", max_length=512, return_tensors="np")

predictions = model.predict({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']})

predicted_class_indices = np.argmax(predictions.logits, axis=1)
class_mapping = {0: 'cannabis', 1: 'non-cannabis', 2: 'non-drugs'}
predicted_classes = [class_mapping[idx] for idx in predicted_class_indices]

df2['predicted_category'] = predicted_classes
df2.to_csv(r'C:\Users\Louis\Downloads\archive (1)\drug_listings_category.csv', encoding='utf-8', index=False)

category_counts = df2['predicted_category'].value_counts()
print(category_counts)
category_percentage = df2['predicted_category'].value_counts(normalize=True) * 100
print(category_percentage)
