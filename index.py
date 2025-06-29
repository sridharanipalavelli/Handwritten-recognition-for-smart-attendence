import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import os

# Step 1: Train or load a CNN model
model_filename = "mnist_cnn_model.h5"

if os.path.exists(model_filename):
    # Load existing model
    model = tf.keras.models.load_model(model_filename)
else:
    # Build CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Load and prepare MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Compile and train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(model_filename)  # Save for next time

class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer (CNN)")
        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        btn_frame = Frame(self)
        btn_frame.pack()

        Button(btn_frame, text="Predict", command=self.predict_digit).grid(row=0, column=0, padx=10)
        Button(btn_frame, text="Clear", command=self.clear_canvas).grid(row=0, column=1)

        self.result_label = Label(self, text="Draw a digit", font=("Helvetica", 24))
        self.result_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (28, 28), color=0)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        # Scale coordinates to 28x28 image
        scale_x = x * 28 // self.canvas_width
        scale_y = y * 28 // self.canvas_height
        self.draw_image.ellipse((scale_x - 1, scale_y - 1, scale_x + 1, scale_y + 1), fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw_image = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit")

    def predict_digit(self):
        img = self.image
        img = ImageOps.invert(img)  # Make background black and digit white
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28)

        prediction = model.predict(img)
        predicted = np.argmax(prediction)

        confidence = np.max(prediction) * 100
        self.result_label.config(text=f"Predicted: {predicted} ({confidence:.2f}%)")
app = DigitRecognizer()
app.mainloop()
