from setuptools import find_packages, setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
      long_description = f.read()

setup(
      name='networks_tracing',
      version='1.0.0',
      description='Agent based covid simulations on networks.',
      author='Felix Gigler',
      author_email='felix.gigler@tuwien.ac.at',
      url='https://github.com/figlerg/network_tracing.git',
      install_requires=['numpy', 'matplotlib', 'networkx', 'scipy', 'ffmpeg-python','pandas','tqdm'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='BSD',
      python_requires='>=3.8',
      packages=find_packages(),
      classifiers=[
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3.8',
      ],
)
