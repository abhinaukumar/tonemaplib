from setuptools import setup, find_packages

setup(
    name='tonemaplib',
    author='Abhinau Kumar',
    author_email='ab.kumr98@gmail.com',
    version='0.1.0',
    url='https://github.com/abhinaukumar/tonemaplib',
    description='Package for HDR video tone mapping in Python.',
    install_requires=[
        'PyWavelets',
        'scikit-image',
        'protobuf',
        'tensorflow',
        'torch',
        'torchvision',
        'joblib',
        'videolib @ git+https://github.com/abhinaukumar/videolib@main',
        'opencv-python',
        'imageio',
        'easydict',
        'scipy',
        'matplotlib',
        'gdown'
    ],
    python_requires='>=3.7.0',
    license='MIT License',
    packages=find_packages()
)
