import setuptools

setuptools.setup(
    name="laughing_net",
    packages=setuptools.find_packages(),
    install_requires=[
        'pytorch-pretrained-bert',
        'tqdm',
        'boto3',
        'requests',
        'regex',
        'pytorch-nlp',
        'sacremoses',
        'sentencepiece',
        'pytorch_transformers',
        'transformers',
        'gensim',
        'spacy',
        'python-box',
        'scikit-learn',
        'keras-preprocessing',
        'pandas',
        'click'
    ],
)
