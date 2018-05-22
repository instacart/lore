from io import open
from setuptools import setup

import sys
sys.lore_no_env = True

import lore


def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()

sql = ['sqlalchemy>=1.2.0b3, <1.2.99', 'sqlalchemy-migrate>=0.11, <0.11.99']
postgres = ['psycopg2>=2.7, <2.7.99'] + sql
redshift = ['sqlalchemy-redshift>=0.7, <0.7.99'] + sql
redis = ['redis>=2.10, <2.10.99']
s3 = ['boto3>=1.4, <1.7.99', 'python-dateutil>=2.1, <2.7.0']
smart_open = ['smart-open>=1.5, <1.5.99'] + s3
geoip = ['geoip2']

keras = [
    'Keras>=2.0.9, <2.1.99',
    'tensorflow>=1.3, <1.5.99',
    'dill>=0.2, <0.2.99',
    'h5py>=2.7, <2.7.99',
    'bleach==1.5.0',
    'html5lib==0.9999999',
    'pydot>=1.2.4, <1.2.99',
    'graphviz>=0.8.2, <0.8.99']
xgboost = ['xgboost>=0.6a2, <0.6.99']
scikit = ['scikit-learn>=0.19, <0.19.99']

all = list(set(keras + xgboost + scikit + postgres + redshift + redis + s3 + geoip))
devel = all + ['moto>=1.1, <1.1.99', 'sphinx', 'sphinx-autobuild', 'sphinx_rtd_theme']

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
        'lore.estimators',
        'lore.estimators.holt_winters',
        'lore.features',
        'lore.io',
        'lore.models',
        'lore.pipelines',
        'lore.stores',
        'lore.www',
        'lore.template.init.app',
        'lore.template.init.app.estimators',
        'lore.template.init.app.models',
        'lore.template.init.app.pipelines',
        'lore.template.init.tests.unit',
    ],
    install_requires=[
        'flask>=0.11.0, <0.12.99',
        'future>=0.15, <0.16.99',
        'inflection>=0.3, <0.3.99',
        'jupyter>=1.0, <1.0.99',
        'jupyter-core>=4.4.0, <4.4.99',
        'numpy>=1.14, <1.14.99',
        'pandas>=0.20, <0.23.99, !=0.22.0',
        'python-dateutil>=2.1, <2.7.0',
        'python-dotenv>=0.6, <0.7.99',
        'six>=1.10, <1.11.99',
        'tabulate>=0.7.5, <0.8.99',
        'shap>=0.12.0, <0.12.99',
    ] + smart_open,
    extras_require={
        'all': all,
        'devel': devel,
        'keras': keras + scikit,
        'postgres': postgres,
        'redis': redis,
        'redshift': redshift,
        's3': s3,
        'scikit': scikit,
        'xgboost': xgboost + scikit,
        'geoip': geoip,
    },
    zip_safe=True,
    test_suite='tests',
    tests_require=[],
    package_data={
        '': [
            'data/names.csv',
            'template/init/app/extracts/*',
            'template/init/config/*',
            'template/init/notebooks/*',
            'template/*',
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
