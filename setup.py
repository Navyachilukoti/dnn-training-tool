from setuptools import setup, find_packages

setup(
    name="dnn-training-tool",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'tensorflow>=2.5.0',
        'matplotlib>=3.3.0',
        'pillow>=8.0.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A GUI tool for training and evaluating Deep Neural Networks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dnn-training-tool",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
) 