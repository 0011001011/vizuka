import os
import shutil
import logging

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

predictor_name = 'predict'
version = '_MNIST_example'
extension = '.npz'

logging.info("Downloading MNIST")
mnist = dt.fetch_mldata('MNIST original')

raw = mnist['data']

logging.info("Preprocessing MNIST")
x = [pr.normalize(x.reshape(1,28**2))[0] for x in raw]
y = [str(int(y)) for y in mnist['target']]


logging.info("Fitting a stupid logistic regression on MNIST")
l = LogisticRegression()
l.fit(x,y)
pred = l.predict(x)


logging.info("Saving all files in vizuka/data/*.npz")
np.savez(
        ''.join([
            os.path.join(DATA_PATH, RAW_NAME),
            version,
            extension,
            ]),
        originals=[[image.reshape(28,28)] for image in raw],
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
        pred=pred,
        )

shutil.copy(
        os.path.join(os.path.dirname(__file__), '2Dembedding1_50_1000_random_12000_MNIST_example.npz'),
        REDUCED_DATA_PATH,
        )
