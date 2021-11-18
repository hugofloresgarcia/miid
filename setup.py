from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='miid',
    version='0.0.1',
    description='Musical Instrument ID utilities',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Hugo Flores Garcia',
    author_email='hugofloresgarcia@u.northwestern.edu',
    keywords=['Audio', 'Dataset', 'PyTorch'],
    url='https://github.com/hugofloresgarcia/miid',
    # download_url='https://pypi.org/project/philharmonia-dataset/'
)

install_requires = [
    'pandas', 
    'torch',
    'torchaudio', 
    'librosa',
    'numpy',
    'pydub', 
    'tqdm',
    'openl3'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)