from setuptools import setup
import os

setup(
        name='Vizuka',
        version='0.25',
        packages=['vizuka/'],
        entry_points = {
            'console_scripts':['vizuka=vizuka.launch_viz:main'],
            },
        description='Represents your high-dimensional datas in a 2D space and play wih it',
        long_description = open(os.path.join(os.path.dirname(__file__),'README.md')).read(),
        install_requires = open(os.path.join(os.path.dirname(__file__), 'vizuka/requirements/requirements.txt')).read(),
        license = 'GPL V3',
        author='Sofian Medbouhi',
        author_email='sof.m.sk@free.fr',
        )
            
