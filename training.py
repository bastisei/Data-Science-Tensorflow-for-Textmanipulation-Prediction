import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model
from collections import Counter

# 1. Load Data
df = pd.read_csv('text_transformation_examples.csv', delimiter='|', quotechar='"')

# 2. Data Cleaning
df.dropna(inplace=True)  # Remove rows with missing values

# 3. Feature and Label Selection
X = df[['Input', 'Output']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
y = df['Transformation']

# 4. Text Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)  # Assume a maximum sequence length of 100

# 5. Label Encoding and One-Hot Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# Print Shapes for Debugging
print("Shape of X_pad:", X_pad.shape)
print("Shape of y_categorical:", y_categorical.shape)

# 6. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_categorical, test_size=0.2, random_state=42)
train_labels = set(y_train.argmax(axis=1))
test_labels = set(y_test.argmax(axis=1))
unseen_labels = test_labels - train_labels
print("Unseen labels in test data:", unseen_labels)

# 7. Model Parameters
vocab_size = len(tokenizer.word_index) + 1  # Total number of unique words
embedding_dim = 50  # Dimension of the embedding vector
num_classes = y_categorical.shape[1]  # Number of classes

# 8. Build Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=100))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Train Model
model.fit(X_train, y_train, epochs=6, batch_size=32, validation_data=(X_test, y_test))

# 10. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 11. Save and Load Model
model.save("transform_model.h5")
print("Model saved successfully.")
loaded_model = load_model("transform_model.h5")
print("Model loaded successfully.")

# 12. Prediction with Loaded Model
predictions_loaded_model = loaded_model.predict(X_pad)
predicted_class_loaded_model = encoder.inverse_transform(predictions_loaded_model.argmax(axis=-1))
# print(f"Predicted Transformation with loaded model: {predicted_class_loaded_model[0]}")

# 13. Function to Predict Transformation Pattern
def predict_transformation(df, column_name, tokenizer, encoder, model, examples):
    # Create a copy of the original DataFrame to make modifications
    predicted_df = df.copy()

    # Replace example values in the specified column
    num_examples = len(examples)
    predicted_df.loc[:num_examples-1, column_name] = examples

    # Tokenize the examples
    examples_seq = tokenizer.texts_to_sequences(predicted_df[column_name][:num_examples].astype(str))
    examples_pad = pad_sequences(examples_seq, maxlen=100)

    # Predict using the model
    predictions = model.predict(examples_pad)

    # Convert predictions into transformation classes
    predicted_classes = encoder.inverse_transform(predictions.argmax(axis=-1))

    # Aggregate the most common pattern
    most_common_pattern = Counter(predicted_classes).most_common(1)[0][0]

    return most_common_pattern

# 14. Test the Pattern Prediction Function
building_permits_df = pd.read_csv('Building_Permits.csv', low_memory=False)
building_permits_df['Permit Number'] = building_permits_df['Permit Number'].astype(str)
loaded_model = load_model("transform_model.h5")
new_data = ['19', '46', '09']

# 15. Make Prediction
predicted_pattern = predict_transformation(building_permits_df, 'Permit Number', tokenizer, encoder, loaded_model, new_data)
print(f"Identified Pattern: {predicted_pattern}")
