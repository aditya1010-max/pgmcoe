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

    2: `# Practical 2 - Single-pass clustering
docs = ["apple banana", "apple fruit", "car bus", "bus train"]
clusters = {}
for doc in docs:
    key = doc.split()[0]
    clusters.setdefault(key, []).append(doc)
print(clusters)`,

    3: `# Practical 3 - Inverted Index
from collections import defaultdict
docs = {'D1': 'apple orange', 'D2': 'orange banana'}
index = defaultdict(list)
for doc, text in docs.items():
    for word in text.split():
        index[word].append(doc)
print(index)`,

    4: `# Practical 4 - Precision and Recall
relevant = {'D1', 'D2', 'D4'}
retrieved = {'D1', 'D2', 'D3'}
precision = len(relevant & retrieved)/len(retrieved)
recall = len(relevant & retrieved)/len(relevant)
print("Precision:", precision, "Recall:", recall)`,

    5: `# Practical 5 - F-measure and E-measure
P, R = 0.5, 0.8
F = (2 * P * R) / (P + R)
E = 1 - F
print("F-measure:", F, "E-measure:", E)`,

    6: `# Practical 6 - Image Feature Extraction (Histogram)
import cv2, matplotlib.pyplot as plt
img = cv2.imread('image.jpg')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.show()`
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
