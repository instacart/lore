from io import open
from setuptools import setup

import sys
sys.lore_no_env = True

import lore


def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()

postgres = ['psycopg2', 'sqlalchemy==1.2.0b3', 'sqlalchemy-migrate']
redis = ['redis']
s3 = ['boto3']

keras = ['keras', 'tensorflow', 'dill']
xgboost = ['xgboost']
scikit = ['scikit-learn']

all = keras + xgboost + scikit + postgres + redis
devel = all + ['moto']

setup(
    name='lore',
    version=lore.__version__,
    description='a framework for building and using data science',
    long_description=readme(),
    classifiers=[
        lore.__status__,
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Environment :: Console',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    keywords='machine learning framework tensorflow airflow',
    url='http://github.com/instacart/lore',
    author=lore.__author__,
    author_email=lore.__email__,
    license='MIT',
    packages=[
        'lore',
        'lore.io',
        'lore.stores',
        'lore.estimators',
        'lore.estimators.holt_winters',
        'lore.features'
    ],
    install_requires=[
        'cookies',
        'cryptography',
        'future',
        'inflection',
        'jsonpickle',
        'jupyter',
        'numpy',
        'pandas',
        'python-dotenv',
        'readline',
        'six',
        'smart-open',
        'wrapt',
        'xmltodict',
    ],
    extras_require={
        'all': all,
        'devel': devel,
        'keras': keras,
        'postgres': postgres,
        'redis': redis,
        's3': s3,
        'scikit': scikit,
        'xgboost': xgboost,
    },
    zip_safe=True,
    test_suite='tests',
    tests_require=[],
    package_data={
        '': [
            'data/names.csv',
            'template/.*',
            'template/**/*',
            'template/**/.*',
            'template/.**/*',
            'template/.**/.*'
        ]
    },
    entry_points={
        'console_scripts': [
            'lore=lore.__main__:main',
        ],
    },
)
