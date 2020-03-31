from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import face_recognition as fr

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/face_rank_model.h5'

# Load your own trained model
model=load_model(MODEL_PATH)
model.summary()
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(dir_path, img_name, ip, model):

    # Preprocessing the image
    whole_path = os.path.join(dir_path, img_name)
    image = fr.load_image_file(whole_path)
    encs = fr.face_encodings(image)
    if len(encs) != 1:
        print("Find %d faces in %s" % (len(encs), whole_path))
        if len(encs) == 0:
            return "Error: no face detected"
        elif len(encs) > 1:
            return "Error: more than one face detected"

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')  wozhushidiaole

    preds = model.predict(np.array(encs))
    print(type(preds))

    # Process your result for human
    score = round(preds[0][0]*2, 3)
    print('score:',score)
    # print(type(score))
    # print(score)

    # Rename img
    new_name = str(score) + "-" + ip + "-" + img_name
    os.rename(os.path.join(dir_path, img_name), os.path.join(dir_path, new_name))

    return "Score is: " + str(score)


@app.route('/', methods=['GET'])
def index():
    # Main page
    # ip = request.remote_addr
    # get real client ip when using ngrok
    ip = request.headers.get("x-forwarded-for")
    print(ip)
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    # ip = request.remote_addr
    # get real client ip when using ngrok
    ip = request.headers.get("x-forwarded-for")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        dir_path = os.path.join(basepath, 'uploads')
        filename = secure_filename(f.filename)
        file_path = os.path.join(dir_path, filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(dir_path, filename, ip, model)
        # print(preds[0])

        return preds
    return None


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()

