import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Setting Constants
VOCAB_SIZE = 10000
MAX_LENGTH = 1000
MODEL_PATH = "./api/models/model.h5"
TRUNC_TYPE ='post'
PADDING_TYPE ='post'
OOV_TOKEN = "<OOV>"

# Load Model & Tokenizer 
model = load_model(MODEL_PATH)
with open('./api/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def encode_text(text):
    text_sequences = tokenizer.texts_to_sequences(text)
    print("**")
    print(len(text_sequences))
    padded_text_sequences = pad_sequences(text_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    return padded_text_sequences

def is_fake(text):
    text_encoded = encode_text(np.array([text]))
    prediction = model.predict(text_encoded)
    prediction = round(prediction[0][0]) 
    print("Prediction: " + str(prediction))
    return prediction
    
