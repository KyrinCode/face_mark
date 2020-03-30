# coding: utf8
import glob
import numpy as np
from keras.models import load_model
import face_recognition as fr

def main():
    model = load_model("face_rank_model.h5")
    files = []
    # files.extend(glob.glob(r"samples/a*"))
    # files.extend(glob.glob(r"samples/b*"))
    # files.extend(glob.glob(r"samples/c*"))
    # files.extend(glob.glob(r"samples/d*"))
    # files.extend(glob.glob(r"samples/e*"))
    # files.extend(glob.glob(r"samples/f*"))
    # files.extend(glob.glob(r"samples/g*"))
    files.extend(glob.glob(r"samples/h*"))
    for f in files:
        image = fr.load_image_file(f)
        # locs = fr.face_locations(image, model="cnn")
        # encs = fr.face_encodings(image, locs)
        encs = fr.face_encodings(image)
        if len(encs) != 1:
            print("Find %d faces in %s" % (len(encs), f))
            continue
        predicted = model.predict(np.array(encs))
        predicted = np.squeeze(predicted)
        # print(type(predicted))
        print("%s: %.4f" % (f, predicted))

if __name__ == '__main__':
    main()