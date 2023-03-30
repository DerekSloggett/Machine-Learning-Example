import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import random
import sys

# Load the data
text = open('text.txt', 'r').read()
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
num_chars = len(chars)

# Prepare the data
max_len = 50
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i:i + max_len])
    next_chars.append(text[i + max_len])

x = np.zeros((len(sentences), max_len, num_chars), dtype=np.bool)
y = np.zeros((len(sentences), num_chars), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Build the model
model = tf.keras.models.Sequential([
    LSTM(128, input_shape=(max_len, num_chars)),
    Dense(num_chars, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')


# Define a function to generate text
def generate_text(epoch, _):
    start_idx = random.randint(0, len(text) - max_len - 1)
    generated_text = text[start_idx:start_idx + max_len]

    sys.stdout.write('\nGenerated text: ' + generated_text)

    for i in range(400):
        x_pred = np.zeros((1, max_len, num_chars))
        for j, char in enumerate(generated_text):
            x_pred[0, j, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_char_idx = np.argmax(preds)
        next_char = idx_to_char[next_char_idx]

        generated_text += next_char
        generated_text = generated_text[1:]

        sys.stdout.write(next_char)
        sys.stdout.flush()


# Define a callback to generate text after each epoch
text_generation_callback = LambdaCallback(on_epoch_end=generate_text)

# Train the model
model.fit(x, y, batch_size=128, epochs=50, callbacks=[text_generation_callback])