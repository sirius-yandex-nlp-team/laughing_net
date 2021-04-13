import setuptools

setuptools.setup(
    name="laughing_net",
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm',
        'boto3',
        'requests',
        'regex',
        'sacremoses',
        'sentencepiece',
        'pytorch-nlp',
        'pytorch-pretrained-bert',
        'pytorch_transformers',
        'pytorch_lightning',
        'transformers',
        'gensim',
        'spacy',
        'python-box',
        'pandas',
        'click',
        'neptune-client',
        'rich',
        'dvc',
        'dvclive'
    ],
)
