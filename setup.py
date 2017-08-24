from setuptools import setup
import os

setup(
        name='Data-viz',
        version='0.11',
        packages=['data_viz/'],
        entry_points = {
            'console_scripts':['vizuka=data_viz.launch_viz:main'],
            },
        description='Represents your high-dimensional datas in a 2D space and play wih it',
        long_description = open(os.path.join(os.path.dirname(__file__),'README.md')).read(),
        install_requires = open('data_viz/requirements/requirements.txt').read(),
        license = 'GPL V3',
        author='Sofian Medbouhi',
        author_email='sof.m.sk@free.fr',
        )
            
