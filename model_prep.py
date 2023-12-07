import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

def load_data(file_path='intents.json'):
    with open(file_path) as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    return padded_sequences, np.array(training_labels), tokenizer, lbl_encoder, num_classes, max_len

def create_model(vocab_size, embedding_dim, max_len, num_classes):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, padded_sequences, training_labels, epochs=500):
    history = model.fit(padded_sequences, training_labels, epochs=epochs)
    return model, history

def save_model(model, model_path='chat_model'):
    model.save(model_path)

def save_tokenizer(tokenizer, file_path='tokenizer.pickle'):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_label_encoder(lbl_encoder, file_path='label_encoder.pickle'):
    with open(file_path, 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    data = load_data()
    padded_sequences, training_labels, tokenizer, lbl_encoder, num_classes, max_len = preprocess_data(data)
    model = create_model(vocab_size=1000, embedding_dim=16, max_len=max_len, num_classes=num_classes)
    trained_model, history = train_model(model, padded_sequences, training_labels)
    save_model(trained_model)
    save_tokenizer(tokenizer)
    save_label_encoder(lbl_encoder)
