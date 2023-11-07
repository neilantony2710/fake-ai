import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
import pickle

#Data Cleaning
'''
originalData = pd.read_csv("./data/original_data.csv")
print("************************8")

# Getting rid of unneccesary columns
data = originalData[["text","label"]].dropna()
print("type:" + str(type(data)))
print(data.shape)

# Changing label values real -> 1, fake -> 0
data.loc[data["label"]=="Real", "label"] = 1 
data.loc[data["label"]=="Fake", "label"] = 0 
print(data.head)

# Writing pandas dataframe to a csv file (makes processing the data into tf data pipeline simpler)
data.to_csv("./data/clean_data.csv")
'''

data = pd.read_csv("./data/clean_data.csv")
data = data.sample(frac=1) # Shuffles Data
text = list(data["text"])
labels = list(data["label"])

# Splitting data into training / validation
# 2050 total items, 80/20 split -> training size of 1640
TRAINING_SIZE = 1640 
text_training = text[:TRAINING_SIZE]
labels_training = labels[:TRAINING_SIZE]
text_testing = text[TRAINING_SIZE:]
labels_testing = labels[TRAINING_SIZE:]

# Toeknizing, Sequencing, and Padding Data
VOCAB_SIZE = 10000
MAX_LENGTH = 1000 
EMBEDDING_DIM = 128 
TRUNC_TYPE ='post'
PADDING_TYPE ='post'
OOV_TOKEN = "<OOV>"

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token = OOV_TOKEN)
tokenizer.fit_on_texts(text_training)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(text_training)
padded_text_training = pad_sequences(training_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

testing_sequences = tokenizer.texts_to_sequences(text_testing)
padded_text_testing = pad_sequences(testing_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

# Saving Tokenizer to a file, for use in the web app
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# Convert to NP Arrays
padded_text_training = np.array(padded_text_training)
labels_training = np.array(labels_training) 
padded_text_testing = np.array(padded_text_testing)
labels_testing = np.array(labels_testing)

# Building Model
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.25),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

model.fit(padded_text_training, labels_training, epochs=1, batch_size=32, validation_split=0.2)

#model.save("model.h5")

loss, accuracy = model.evaluate(padded_text_testing, labels_testing)
print(f"Test accuracy: {accuracy * 100:.2f}%")

prediction = model.predict(np.array([padded_text_testing[1]]))
print(prediction)
print(len(prediction))
print("Excpeced: " +str(labels_testing[1]))
