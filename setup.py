from io import open
from setuptools import setup

import builtins
builtins.lore_no_env = True

import lore


def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name='lore',
    version=lore.__version__,
    description='a framework for building and using data science',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    keywords='machine learning framework tensorflow airflow',
    url='http://github.com/instacart/lore',
    author='Montana Low',
    author_email='montana@instacart.com',
    license='MIT',
    packages=[
        'lore',
        'lore.io',
        'lore.stores',
        'lore.estimators',
        'lore.estimators.holt_winters',
    ],
    install_requires=[
        'future',
        'smart-open',
        'keras',
        'tensorflow',
        'scikit-learn',
        'h5py',
        'numpy',
        'pandas',
        'python-dotenv',
        'jupyter',
        'dill',
        'inflection',
        'xgboost',
        'tabulate',
        'sqlalchemy',
        'six',
    ],
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
