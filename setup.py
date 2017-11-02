from io import open
from setuptools import setup

import sys
sys.lore_no_env = True

import lore


def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()

postgres = ['psycopg2>=2.7, <2.8', 'sqlalchemy>=1.2.0b3, <1.3', 'sqlalchemy-migrate>=0.11, <0.12']
redis = ['redis>=2.10, <2.11']
s3 = ['boto3>=1.4, <1.5']

keras = ['Keras==2.0.9', 'tensorflow>=1.4, <1.5', 'dill>=0.2, <0.3', 'h5py>=2.7, <2.8']
xgboost = ['xgboost>=0.6a2, <0.7']
scikit = ['scikit-learn>=0.19, <0.20']

all = keras + xgboost + scikit + postgres + redis
devel = all + ['moto>=1.1,<1.2']

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
        'future>=0.16, <0.17',
        'inflection>=0.3, <0.4',
        'jupyter>=1.0, <1.1',
        'numpy>=1.13, <1.14',
        'pandas>=0.21, <0.22',
        'python-dotenv>=0.6, <0.7',
        'six>=1.11, <1.12',
        'smart-open>=1.5, <1.6',
        'tabulate>=0.8, <0.9',
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
