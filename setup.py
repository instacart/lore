from io import open
from setuptools import setup

import sys
sys.lore_no_env = True

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
        'lore.tasks',
        'lore.www',
        'lore.template.init.app',
        'lore.template.init.app.estimators',
        'lore.template.init.app.models',
        'lore.template.init.app.pipelines',
        'lore.template.init.tests.unit',
    ],
    zip_safe=True,
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
