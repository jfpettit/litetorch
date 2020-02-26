from setuptools import setup

with open("README.md", "r") as fh:
	long_description = fh.read()

setup(
	name='litetorch',
	version='0.1.0',
	author='Jacob Pettit',
	author_email='jfpettit@gmail.com',
    short_description='Lightweight model classes on top of PyTorch to reduce rewriting of common models.',
    long_description=long_description,
	install_requires=['numpy', 'torch', 'scipy', 'torchtext', 'termcolor', 'joblib', 'typing', 'spacy']
	,
)