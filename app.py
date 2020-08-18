# coding=utf-8
import tensorflow as tf
import numpy as np
import keras
import os
import time

# SQLite for information
import sqlite3

# Keras
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from keras.utils.data_utils import get_file
from PIL import Image

# Flask utils
from flask import Flask, url_for, render_template, request,send_from_directory,redirect
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# load json file before weights
loaded_json = open("models/cars.json", "r")
# read json architecture into variable
loaded_json_read = loaded_json.read()
# close file
loaded_json.close()
# retreive model from json
loaded_model = model_from_json(loaded_json_read)
# load weights
weights_path = get_file(
        'car_weights.h5',
        'https://project3cars.s3.us-east-2.amazonaws.com/model.h5')
loaded_model.load_weights(weights_path)


def info():
    conn = sqlite3.connect("models/cars196.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Cars")
    rows = cursor.fetchall()
    return rows

def model_predict(img_path):
    # load image with target size
    img = image.load_img(img_path, target_size=(256, 256))
    # convert to array
    img = image.img_to_array(img)
    # normalize the array
    img /= 255
    # expand dimensions for keras convention
    img = np.expand_dims(img, axis=0)

    with tf.Graph().as_default() as g:
        opt = keras.optimizers.Adam(lr=0.001)
        loaded_model.compile(
            optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        preds = loaded_model.predict_classes(img)
        return int(preds)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)
        # Make prediction
        preds = model_predict(img_path)
        rows = info()
        res = np.asarray(rows[preds])
        value = (preds == int(res[0]))
        if value:
            Class, Label = [i for i in res]
            return render_template('result.html', Class=Class, result=Label, filee=f.filename)
        return result
    return None

@app.route('/predict/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)



if __name__ == '__main__':
    app.run()