from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import tensorflow as tf  # Import TensorFlow module
import pickle
from model_prep import load_data, preprocess_data, create_model, save_model, save_tokenizer, save_label_encoder

app = Flask(__name__)

# Load the trained model, tokenizer, and label encoder
model = tf.keras.models.load_model('chat_model')  # Use tf.keras.models instead of keras.models

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Load intents data
data = load_data()

def get_bot_response(user_input):
    # Use the provided example inference code here
    tokenized_input = tokenizer.texts_to_sequences([user_input])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(tokenized_input, truncating='post', maxlen=20)
    result = model.predict(padded_input)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

    return "Sorry, I didn't understand that."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.get_json()['user_input']
    bot_response = get_bot_response(user_input)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
