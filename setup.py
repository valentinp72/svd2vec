from setuptools import setup

setup(name='svd2vec',
      version='0.1',
      description='A library that converts words to vectors using PMI and SVD',
      url='https://git-lium.univ-lemans.fr/vpelloin/svd2vec',
      author='Valentin Pelloin',
      author_email='valentin.pelloin.etu@univ-lemans.fr',
      license='MIT',
      packages=['svd2vec'],
      package_data={'svd2vec': ['datasets/similarities/*.txt', 'datasets/analogies/*.txt']},
      zip_safe=False)
