from setuptools import setup, find_packages
import os

data = {
        'vizuka':
        [
            'data/set/.gitkeep',
            'data/models/.gitkeep',
            'data/cache/.gitkeep',
            'data/saved_clusters/.gitkeep',
            'data/reduced/.gitkeep',
            'example/tsne#MNIST_example#learning_rate::1000#n_iter::12000#perplexity::50.npz',
        ],
    }

setup(
        name='Vizuka',
        version='0.32.0',
        packages=find_packages(),#['vizuka/'],
        package_data = data,
        entry_points = {
            'console_scripts':[
                'vizuka=vizuka.launch_viz:main',
                'vizuka-reduce=vizuka.launch_reduce:main',
                ],
            },
        description='Represents your high-dimensional datas in a 2D space and play with it',
        long_description = open(os.path.join(os.path.dirname(__file__),'README.md')).read(),
        install_requires = open(os.path.join(os.path.dirname(__file__), 'vizuka/requirements/requirements.txt')).read(),
        license = 'GPL V3',
        author='Sofian Medbouhi',
        author_email='sof.m.sk@free.fr',
        )
            
