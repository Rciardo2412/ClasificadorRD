from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo solo una vez
model = tf.keras.applications.InceptionResNetV2(include_top=True, weights='imagenet')

# Función para preprocesar la imagen
def preprocess_image(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img

# Función para clasificar la imagen
def classify_image(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_classes = tf.keras.applications.inception_resnet_v2.decode_predictions(predictions, top=5)[0]
    return predicted_classes

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = file.read()
            try:
                predicted_classes = classify_image(image)
                return render_template('result.html', predictions=predicted_classes)
            except Exception as e:
                return render_template('upload.html', error=str(e))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

