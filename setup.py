from setuptools import find_packages, setup

VERSION = "0.0.6"
DESCRIPTION = "Representational Similarity Measures for Deep Learning"
LONG_DESCRIPTION = "DeepReps is a comprehensive library designed to centralize various measures of similarity for deep learning representations. It is specifically tailored for comparing the weights and outputs of deep learning models, providing a valuable toolkit for researchers and practitioners to assess and analyze model similarities."


setup(
    name="deep_reps",
    version=VERSION,
    author="Mezosky",
    author_email="<imezadelajara@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "torch>=2.3.1",
        "matplotlib>=3.6.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "plotly>=5.11.0",
    ],
    keywords=["python", "deep-learning", "torch", "similarity", "representations"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
