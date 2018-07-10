DATEUTIL = ['python-dateutil>=2.1, <2.7.0']
FLASK = ['flask>=0.11.0, <0.12.99']
FUTURE = ['future>=0.15, <0.16.99']
INFLECTION = ['inflection>=0.3, <0.3.99']
JINJA = ['Jinja2>=2.9.0, <2.10.0']
JUPYTER = [
    'jupyter>=1.0, <1.0.99',
    'jupyter-core>=4.4.0, <4.4.99',
]
NUMPY = ['numpy>=1.14, <1.14.99']
PANDAS = ['pandas>=0.20, <0.23.99, !=0.22.0']
TABULATE = ['tabulate>=0.7.5, <0.8.99']
SHAP = ['shap>=0.12.0, <0.12.99']

SQL = ['sqlalchemy>=1.2.0b3, <1.2.99', 'sqlalchemy-migrate>=0.11, <0.11.99']
SNOWFLAKE = [
    'snowflake-connector-python>=1.5.5, <1.6.0',
    'snowflake-sqlalchemy>=1.1.0, <1.2.0',
]
POSTGRES = ['psycopg2>=2.7, <2.7.99'] + SQL
REDSHIFT = ['sqlalchemy-redshift>=0.7, <0.7.99'] + SQL
REDIS = ['redis>=2.10, <2.10.99']
S3 = ['boto3>=1.4, <1.7.99'] + DATEUTIL
SMART_OPEN = ['smart-open>=1.5, <1.5.99'] + S3
GEOIP = ['geoip2']
H5PY = ['h5py>=2.7, <2.8.99',]
KERAS = [
    'Keras>=2.0.9, <2.1.99',
    'tensorflow>=1.3, <1.5.99',
    'dill>=0.2, <0.2.99',
    'bleach==1.5.0',
    'html5lib==0.9999999',
    'pydot>=1.2.4, <1.2.99',
    'graphviz>=0.8.2, <0.8.99'] + H5PY
XGBOOST = ['xgboost>=0.72, <0.80']
SKLEARN = ['scikit-learn>=0.19, <0.19.99']

ALL = list(set(
    DATEUTIL +
    FLASK +
    FUTURE +
    INFLECTION +
    JINJA +
    JUPYTER +
    NUMPY +
    PANDAS +
    TABULATE +
    SHAP +
    SQL +
    SNOWFLAKE +
    POSTGRES +
    REDSHIFT +
    REDIS +
    S3 +
    SMART_OPEN +
    GEOIP +
    H5PY +
    KERAS +
    XGBOOST +
    SKLEARN
))

TEST = ALL + [
    'moto>=1.1, <1.3.99'
]

DOC = ALL + [
    'sphinx',
    'sphinx-autobuild',
    'sphinx_rtd_theme'
]
