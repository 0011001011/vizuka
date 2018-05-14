import os
import shutil

import sklearn.datasets as dt
import sklearn.preprocessing as pr
from sklearn.linear_model import LogisticRegression

import numpy as np

from vizuka.config import (
        DATA_PATH,
        MODEL_PATH,
        RAW_NAME,
        INPUT_FILE_BASE_NAME,
        REDUCED_DATA_PATH,
        )

predictor_name = 'predict_'
version = 'MNIST_example'
extension = '.npz'

print("mnist=Downloading")
mnist = dt.fetch_mldata('MNIST original')

raw = mnist['data']

print("mnist=Preprocessing")
x = [pr.normalize(x.reshape(1,28**2))[0] for x in raw]
y = [str(int(y)) for y in mnist['target']]


print("mnist=Fitting a stupid logistic regression")
l = LogisticRegression()
l.fit(x,y)
predictions = l.predict(x)


print("mnist=Saving all files in vizuka/data/*.npz")
np.savez(
        ''.join([
            os.path.join(DATA_PATH, RAW_NAME),
            version,
            extension,
            ]),
        x=[[image.reshape(28,28)] for image in raw],
        columns=['image'],
        )

np.savez(
        ''.join([
            os.path.join(DATA_PATH, INPUT_FILE_BASE_NAME),
            version,
            extension,
            ]),
        x=x,
        y=y,
        )

np.savez(
        ''.join([
            os.path.join(MODEL_PATH, predictor_name),
            version,
            extension,
            ]),
        y=predictions,
        )

shutil.copy(
        os.path.join(os.path.dirname(__file__), 'tsne#MNIST_example#learning_rate@@1000#n_iter@@12000#perplexity@@50.npz'),
        REDUCED_DATA_PATH,
        )
