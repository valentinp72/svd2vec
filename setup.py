from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='svd2vec',
      version='0.2',
      description='A library that converts words to vectors using PMI and SVD',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/valentinp72/svd2vec',
      author='Valentin Pelloin',
      author_email='valentin.pelloin.etu@univ-lemans.fr',
      license='MIT',
      packages=['svd2vec'],
      package_data={'svd2vec': ['datasets/similarities/*.txt', 'datasets/analogies/*.txt']},
      classifiers=[
                  "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: MIT License",
                  "Operating System :: OS Independent",]
      )
