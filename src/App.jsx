import React, { useState } from "react";

function App() {

  const [copied, setCopied] = useState(null);

  // Store all 6 practical codes directly in frontend
  const codes = {
    1: `import numpy as np

import tensorflow as tf
print(tf.__version__)


from keras import datasets
# Load MNIST datasets from keras
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images.shape

test_images.shape


import theano.tensor as T
from theano import function


# Declaring 2 variables
x = T.dscalar('x')
y = T.dscalar('y')

# Summing up the 2 numbers
z = x + y

# Converting it to a callable object so that it takes matrix as parameters
f = function([x, y], z)

f(5, 7)


import torch
import torch.nn as nn

print(torch.__version__)


torch.cuda.is_available()`,

    2: `#imporƟng necessary libraries
import tensorflow as ƞ
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 

%matplotlib inline
#Loading and preparing the data
mnist = ƞ.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#to see length of training dataset
len(x_train)
#shape of training dataset 60,000 images having 28*28 size
x_train.shape
#shape of tesƟng dataset 10,000 images having 28*28 size
x_test.shape
x_train[0]
#to see how first image look
plt.matshow(x_train[1])
#normalize the images by scaling pixel intensiƟes to the range 0,1
x_train = x_train / 255
x_test = x_test / 255
x_train[0]
#Define the network architecture using Keras
#CreaƟng the model
from keras.models import SequenƟal
from keras.layers import Input, FlaƩen, Dense
model = SequenƟal([
 Input(shape=(28, 28)),
 FlaƩen(),
 Dense(64, acƟvaƟon='relu'),
 Dense(10, acƟvaƟon='soŌmax')
])
model.summary()
#Compile the model
model.compile(opƟmizer='sgd',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
#Train the model
history=model.fit(x_train, y_train,validaƟon_data=(x_test,y_test),epochs=10)
#Evaluate the model
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)

print("Accuracy=%.3f" %test_acc)
#Making PredicƟon on New Data
n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()
#we use predict() on new data
predicted_value=model.predict(x_test)
print("HandwriƩen number in the image is= %d" %np.argmax(predicted_value[n]))
#Plot graph for Accuracy and Loss
history.history
history.history.keys()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.Ɵtle('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'ValidaƟon'], loc='upper leŌ')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.Ɵtle('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'ValidaƟon'], loc='upper leŌ')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.Ɵtle('Training Loss and accuracy')
plt.ylabel('accuracy/Loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy','loss','val_loss'])
plt.show()
#Save the model
keras_model_path = 'C:\\Users\\Pushpak Warke\\my_model.keras'
model.save(keras_model_path)
restored_keras_model = ƞ.keras.models.load_model('C:\\Users\\Pushpak Warke\\my_model.keras') `,

    3: `#ImporƟng libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as ƞ
import PIL 
 

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import SequenƟal


import tensorflow as ƞ
import pathlib
dataset_url =
"hƩps://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = ƞ.keras.uƟls.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir) / "flower_photos" # 踬踭踮踯 Fix: go one level deeper
print("Data directory:", data_dir) 


#Number of images in dataset
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count) 


#Roses
roses=list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0])) 


PIL.Image.open(str(roses[1])) 


#Tulips
tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0])) 


PIL.Image.open(str(tulips[1])) 


batch_size = 32
img_height = 180
img_width = 180


train_ds = ƞ.keras.uƟls.image_dataset_from_directory(
 data_dir,
 validaƟon_split=0.2,
 subset="training",
 seed=123,
 image_size=(img_height, img_width),
 batch_size=batch_size) 


val_ds = ƞ.keras.uƟls.image_dataset_from_directory(
 data_dir,
 validaƟon_split=0.2,
 subset="validaƟon",
 seed=123,
 image_size=(img_height, img_width),
 batch_size=batch_size) 


class_names = train_ds.class_names
print(class_names) 


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
 for i in range(9):
 ax = plt.subplot(3, 3, i + 1)
 plt.imshow(images[i].numpy().astype("uint8"))
 plt.Ɵtle(class_names[labels[i]])
 plt.axis("off") 


for image_batch, labels_batch in train_ds:
 print(image_batch.shape)
 print(labels_batch.shape)
 break 


AUTOTUNE = ƞ.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) 


normalizaƟon_layer = layers.Rescaling(1./255) 


normalized_ds = train_ds.map(lambda x, y: (normalizaƟon_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# NoƟce the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


num_classes = len(class_names)
model = SequenƟal([
 layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
 layers.Conv2D(16, 3, padding='same', acƟvaƟon='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', acƟvaƟon='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', acƟvaƟon='relu'),
 layers.MaxPooling2D(),
 layers.FlaƩen(),
 layers.Dense(128, acƟvaƟon='relu'),
 layers.Dense(num_classes)
]) 


model.compile(opƟmizer='adam',
 loss=ƞ.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy']) 


model.summary() 


epochs=10
history = model.fit(
 train_ds,
 validaƟon_data=val_ds,
 epochs=epochs
) 


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='ValidaƟon Accuracy')
plt.legend(loc='lower right')
plt.Ɵtle('Training and ValidaƟon Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='ValidaƟon Loss')
plt.legend(loc='upper right') 
plt.Ɵtle('Training and ValidaƟon Loss')
plt.show()

data_augmentaƟon = keras.SequenƟal(
 [
 layers.RandomFlip("horizontal",
 input_shape=(img_height,
 img_width,
 3)),
 layers.RandomRotaƟon(0.1),
 layers.RandomZoom(0.1),
 ]
) 


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
 for i in range(9):
 augmented_images = data_augmentaƟon(images)
 ax = plt.subplot(3, 3, i + 1)
 plt.imshow(augmented_images[0].numpy().astype("uint8"))
 plt.axis("off") 


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
 for i in range(9):
 augmented_images = data_augmentaƟon(images)
 ax = plt.subplot(3, 3, i + 1)
 plt.imshow(augmented_images[0].numpy().astype("uint8"))
 plt.axis("off")

model.compile(opƟmizer='adam',
 loss=ƞ.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy']) 


model.summary() 


epochs = 15
history = model.fit(
 train_ds,
 validaƟon_data=val_ds,
 epochs=epochs
) `,

    4: `# Import Required Libraries
import numpy as np
from sklearn.model_selecƟon import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
#Step 1: Create SyntheƟc Data
# Normal data (mean=0, std=1)
normal_data = np.random.normal(0, 1, (1000, 10))
# Anomaly data (mean=5, std=1)
anomaly_data = np.random.normal(5, 1, (250, 10))
# Combine both into one dataset
data = np.vstack([normal_data, anomaly_data])
labels = np.hstack([np.zeros(1000), np.ones(250)]) # 0 = normal, 1 = anomaly
#Step 2: Split into Training and TesƟng Sets
x_train, x_test, y_train, y_test = train_test_split(
 data, labels, test_size=0.3, random_state=42
)
#Step 3: Define Autoencoder Architecture
input_dim = x_train.shape[1] # number of features = 10
encoding_dim = 3 # compressed latent dimension
# Input Layer
input_layer = Input(shape=(input_dim,))
# Encoder Layer
encoder_layer = Dense(encoding_dim, acƟvaƟon='relu')(input_layer)
# Decoder Layer
decoder_layer = Dense(input_dim, acƟvaƟon='linear')(encoder_layer)
# Build the Autoencoder Model
autoencoder = Model(input_layer, decoder_layer)
autoencoder.compile(opƟmizer='adam', loss='mse')
#Step 4: Train Autoencoder (Only on Normal Data)
autoencoder.fit(
 x_train[y_train == 0],
 x_train[y_train == 0],
 epochs=10
)
#Step 5: Predict and Detect Anomalies
x_pred = autoencoder.predict(x_test)
mse = np.mean((x_test - x_pred) ** 2, axis=1) 

#Step 6: Calculate Dynamic Threshold using traning normal data
x_train_pred = autoencoder.predict(x_train[y_train == 0])
mse_train = np.mean((x_train[y_train == 0] - x_train_pred) ** 2, axis=1)
threshold = np.mean(mse_train) + 3 * np.std(mse_train) # Dynamic threshold
#Step 7: Predict Anomalies Using Dynamic Threshold
y_pred = (mse > threshold).astype(int) # 1 = anomaly, 0 = normal
#Step 8: Display results
print("Calculated Threshold:", threshold)
print("True labels (first 20):", y_test[:20])
print("Predicted labels (first 20):", y_pred[:20])
print("MSE (first 20 samples):", mse[:20]) `,

    5: `#Step 1: Import Libraries
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.uƟls import to_categorical
import tensorflow.keras.backend as K
#Step 2: Text Data
text = "I love playing cricket and watching cricket matches"
# Tokenizer: convert words into numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word2idx = tokenizer.word_index
vocab_size = len(word2idx) + 1
seq = tokenizer.texts_to_sequences([text])
#Step 3: Create Context–Target Pairs
pairs = []
window = 2 # context words around target word
for i, target in enumerate(seq[0]):
 for j in range(max(0, i - window), min(len(seq[0]), i + window + 1)):
 if i != j:
 pairs.append((seq[0][j], target))
contexts = np.array([x[0] for x in pairs])
targets = np.array([x[1] for x in pairs]) 

targets = to_categorical(targets, vocab_size) # one-hot encode targets
#Step 4: Create CBOW Model
from keras.models import Model
from keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D
input_layer = Input(shape=(1,))
embedding_layer = Embedding(vocab_size, 8, name="embedding")(input_layer)
x = GlobalAveragePooling1D()(embedding_layer) # 膆 replaces Lambda
output_layer = Dense(vocab_size, acƟvaƟon='soŌmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(opƟmizer='adam', loss='categorical_crossentropy')
model.summary()
#Step 5: Train Model
model.fit(contexts, targets, epochs=10)
#Step 6: Predict a Word
test_word = "playing"
test_idx = np.array([[word2idx[test_word]]])
pred = model.predict(test_idx)
predicted_idx = np.argmax(pred)
for w, i in word2idx.items():
 if i == predicted_idx:
 predicted_word = w
 break
print(f"Context word: '{test_word}'")
print(f"Predicted target word: '{predicted_word}'") `,

//     6: `# Practical 6 - Image Feature Extraction (Histogram)
// import cv2, matplotlib.pyplot as plt
// img = cv2.imread('image.jpg')
// colors = ('b', 'g', 'r')
// for i, col in enumerate(colors):
//     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
//     plt.plot(hist, color=col)
// plt.show()`
  };

  // Copy selected code
  const handleCopy = async (id) => {
    await navigator.clipboard.writeText(codes[id]);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#f5f5f5",
        fontFamily: "monospace",
      }}
    >
      <h1 style={{ color: "#222", marginBottom: "30px" }}>
        Practical Code Copier
      </h1>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, 160px)",
          gap: "15px",
        }}
      >
        {[1, 2, 3, 4, 5, 6].map((num) => (
          <button
            key={num}
            onClick={() => handleCopy(num)}
            style={{
              padding: "12px",
              borderRadius: "8px",
              border: "1px solid #ccc",
              backgroundColor: "#fff",
              cursor: "pointer",
              fontSize: "15px",
              color: "#333",
            }}
            onMouseEnter={(e) => (e.target.style.backgroundColor = "#eaeaea")}
            onMouseLeave={(e) => (e.target.style.backgroundColor = "#fff")}
          >
            {copied === num ? "✅ Copied!" : `Practical ${num}`}
          </button>
        ))}
      </div>
    </div>
  );
}

export default App;
