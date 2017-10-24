import os

import numpy as np

from vizuka.config import (
        VERSION,
        REDUCED_DATA_NAME,
        REDUCED_DATA_PATH,
        )

class Projector():

        def __init__(self):
            """
            Instanciate a Projector engine, you should give it
            here all the parameters
            """
            self.projections = np.array([])
            self.parameters = [] # In your implementation, do add you parameters here !
                                 # If you don't, two projections with different
                                 # parameters will be saved in the same file, the new
                                 # one overwritting the old one ..see: self.save
        
        def project(self, x):
            """
            Projects the x vector into a 2D space according to the
            parameters specified in __init__
            """
            self.projections = np.array([[0,0] for _ in range(len(x))])
            return self.projections

        def get_savename(self, version, filename, path):
            name = ''.join('_' + str(param) for param in self.parameters)
            return ''.join([path, filename, name, '_', version, '.npz'])
     
        def load_projection(self, version=VERSION, base_filename=REDUCED_DATA_NAME, path=REDUCED_DATA_PATH):
            full_path = self.get_savename(version, base_filename, path)
            if os.path.exists(full_path):
                return np.load(full_path)['x_2D']
            else:
                import logging
                logging.info("Nothing found in {}!".format(full_path))
                return np.array([])

        def save_projection(self, version=VERSION, base_filename=REDUCED_DATA_NAME, path=REDUCED_DATA_PATH):
            """
            Save the projection in standard location with appropriate name,
            which depends on the parameters of the algorithm
            """
            full_path = self.get_savename(version, base_filename, path)
            np.savez(full_path, x_2D=self.projections)
