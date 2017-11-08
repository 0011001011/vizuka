import os
import logging

import numpy as np

from vizuka.config import (
        VERSION,
        REDUCED_DATA_PATH,
        NAME_VALUE_SEPARATOR,
        PARAMETERS_SEPARATOR,
        )

class Projector():

        def __init__(self):
            """
            Instanciate a Projector engine, you should give it
            here all the parameters
            """
            self.method='null'
            self.projections = np.array([])
            self.parameters = {} # In your implementation, do add you parameters here !
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

        # GENERIC PROJECTOR FUNCTIONS
        #############################

        def register_parameters(self, parameters):
            """
            This function looks silly but it enforces the fact that parameters should
            alml remain in the same dict for some other functions to be able to fetch
            them by name:value
            """
            self.parameters = parameters

        def get_param_from_name(name):
            """
            Returns the method and parameters of the projector
            from its name
            """
            params  = {}
            blocks  = name.split(PARAMETERS_SEPARATOR)
            method, version, *blocks = blocks
            
            for block in blocks:
                if block:
                    name, value  = block.split(NAME_VALUE_SEPARATOR)
                    params[name] = value
            
            return method, version, params

        def get_savename(self, version=VERSION, path=REDUCED_DATA_PATH):
            name = ''.join([self.method, PARAMETERS_SEPARATOR, version])
            ordered_name = np.sort(list(self.parameters.keys()))
            name += ''.join(PARAMETERS_SEPARATOR + str(param_name) + NAME_VALUE_SEPARATOR + str(self.parameters[param_name]) for param_name in ordered_name)
            return ''.join([path, name, '.npz'])
     
        def load_projection(self, version=VERSION, path=REDUCED_DATA_PATH):
            full_path = self.get_savename(version, path)
            if os.path.exists(full_path):
                return np.load(full_path)['x_2D']
            else:
                logging.info("Nothing found in {}!".format(full_path))
                return np.array([])

        def save_projection(self, version=VERSION, path=REDUCED_DATA_PATH):
            """
            Save the projection in standard location with appropriate name,
            which depends on the parameters of the algorithm
            """
            full_path = self.get_savename(version, path)
            np.savez(full_path, x_2D=self.projections)
