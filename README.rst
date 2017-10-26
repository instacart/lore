.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

======
 Lore
======

|circleci_badge|


Lore is a python data science framework to design, fit, and exploit machine learning models from development to production. It codifies best practices to simplify collaborating and deploying models developed on a laptop with Jupyter notebook, into high availability distributed production data centers.


Why?
----
Writing code with a fast feedback loop is fulfilling. With complex data, you can spend hours, days, then weeks iterating through more complex edge cases on larger samples until the last bits are smoothed over. Instead of spending time partially reimplementing common patterns, frequent challenges should be solved once, and thoroughly.



Create a Lore project
--------------------

.. code-block:: bash

  $ pip install lore
  $ lore init my_project --python_version=3.6.3

  # fix up .env, config/database.cfg, circle.yml, README.rst


A Cute Little Example
---------------------

We'll naively try to predict whether users are subscribers, given their first name.



Update config/database.cfg to specify your database url:

.. code-block:: ini

  # config/database.cfg

  [MAIN]
  url: $DATABASE_URL

you can set environment variable for only the lore process with the .env file:

.. code-block:: bash

  # .env

  DATABASE_URL=postgres://user:password@localhost:5432/main_development


Create a sql file that specifies your data:

.. code-block:: sql

  -- app/extracts/subscribers.sql

  SELECT
    first_name,
    has_subscription
  FROM users
  LIMIT = %(limit)s

Pipelines are the unsexy, but essential component of most machine learning applications. They transform raw data into encoded training (and prediction) data for a model. Lore has several features to make data munging more palatable.

.. code-block:: python

  # app/pipelines/subscribers.py

  import lore.io
  import lore.pipelines
  from lore.encoders import Norm, Discrete, Boolean, Unique
  from lore.transformers import NameAge, NameSex, Log


  class TrainTestSplit(lore.pipelines.TrainTestSplit):

      def get_data(self):
          # lore.io.main is a Connection created by config/database.cfg + DATABASE_URL
          # dataframe() supports keyword args for interpolation (limit)
          # subscribers is the name of the extract
          # cache=True enables LRU query caching
          return lore.io.main.dataframe('subscribers', limit=100, cache=True)

      def get_encoders(self):
          # An arbitrairily chosen set of encoders (w/ transformers)
          # that reference sql columns in the extract by name.
          # A fair bit of thought will probably go into expanding
          # your list with features for your model.
          return (
              Unique('first_name', minimum_occurrences=100),
              Norm(Log(NameAge('first_name'))),
              Discrete(NameSex('first_name'), bins=10),
          )

      def get_output_encoder(self):
          # A single encoder that references the predicted outcome
          return Boolean('has_subscription')


The superclass :python:`lore.pipelines.TrainTestSplit` will take care of:

# splitting the data into training_data/validation_data/test_data dataframes
# fitting the encoders to training_data
# transforming training_data/validation_data/test_data for the model

Define some models that will fit and predict the data. Base models are designed to be extended and overridden, but work with defaults out of the box.

.. code-block:: python

  # app/models/subscribers.py

  import lore.models
  from app.pipelines.subscribers import TrainTestSplit

  class DeepName(lore.models.Keras):
      def __init__():
          super(DeepName, self).__init__(
              pipeline=TrainTestSplit(),
              estimator=lore.estimators.Keras() # a canned estimator for deep learning
          )

  class BoostedName(lore.models.Base):
      def __init__():
          super(XGBoostedName, self).__init__(
              pipeline=TrainTestSplit(),
              estimator=lore.estimators.XGBoost() # a canned estimator for XGBoost
          )


Test the models predictive power:

.. code-block:: python

  # tests/unit/subscribers.py

  from app.models.subscribers import DeepName, BoostedName

  class TestSubscribers(unittest.TestCase):
      def test_deep_name(self):
          model = DeepName() # initialize a new model
          model.fit(epochs=20) # fit to the pipeline's training_data
          predictions = model.predict(model.pipeline.test_data.x) # predict the holdout
          self.assertEqual(predictions, model.pipeline.test_data.y) # hah!

      def test_xgboosted_name(self):
          model = BoostedName()
          model.fit()
          predictions = model.predict(model.pipeline.test_data.x)
          self.assertEqual(predictions, model.pipeline.test_data.y) # hah hah hah!

Run tests:

.. code-block:: bash

  $ lore test

Experiment and tune :bash:`notebooks/` with :bash:`$ lore notebook` using the app kernel


Project Structure
-----------------

.. code-block::

  ├── .env.template            <- Template for environment variables for developers (mirrors production)
  ├── README.md                <- The top-level README for developers using this project.
  ├── requirements.txt         <- keeps dev and production in sync (pip)
  ├── runtime.txt              <- keeps dev and production in sync (pyenv)
  │
  ├── data/                    <- query cache and other temp data
  │
  ├── docs/                    <- generated from src
  │
  ├── logs/                    <- log files per environment
  │
  ├── models/                  <- local model store from fittings
  │
  ├── notebooks/               <- explorations of data and models
  │       └── my_exploration/
  │            └── exploration_1.ipynb
  │
  ├── appname/                 <- python module for appname
  │   ├── __init__.py          <- loads the various components (makes this a module)
  │   │
  │   ├── api/                 <- external entry points to runtime models
  │   │   └── my_endpoint.py   <- hub endpoint for predictions
  │   │
  │   ├── extracts/            <- sql
  │   │   └── my_sql.sql
  │   │
  │   ├── estimators/          <- Code that make predictions
  │   │   └── my_estimator.py  <- Keras/XGBoost implementations
  │   │
  │   ├── models/              <- Combine estimator(s) w/ pipeline(s)
  │   │   └── my_model.py
  │   │
  │   └── pipelines/           <- abstractions for processing data
  │       └── my_pipeline.py   <- train/test/split data encoding
  │
  └── tests/
      ├── data/                <- cached queries for fixture data
      ├── models/              <- model store for test runs
      └── unit/                <- unit tests


Modules Overview
================
Lore provides python modules to standardize Machine Learning techniques across multiple libraries.

Core Functionality
------------------
- **lore.models** are compatibility wrappers for your favorite library (keras, xgboost, scikit). They come with reasonable defaults for rough draft training out of the box.
- **lore.pipelines** fetch, encode, and split data into training/test sets for models. A single pipeline will have one Encoder per feature in the model.
- **lore.encoders** operate within Pipelines to transform a single feature into an optimal representation for learning.
- **lore.transformers** provide common operations, like extracting the area code from a free text phone number. They can be chained together inside encoders. They efficiently

Supporting functionality
------------------------
- **lore.io** allows connecting to postgres/redshift and upload/download from s3
- **lore.serializers** persist models with their pipelines and encoders (and get them back again)
- **lore.stores** save intermediate data, for reproducibility and efficiency.

Utilities
---------
- **lore.util** has those extra niceties we rewrite in every project, and then some
- **lore.env** takes care of ensuring that all dependencies are correctly installed before running

Features
========

Integrated Libraries
--------------------
Use your favorite library in a lore project, just like you'd use them in any other python project. They'll play nicely together.

- Keras/Tensorflow + Tensorboard
- XGBoost
- Scikit-Learn
- Jupyter Notebook
- Pandas
- Numpy
- Matplotlib, ggplot, plotnine
- Sqlalchemy, Psycopg2
- Hub

Dev Ops
-------
There are many ways to manage python dependencies in development and production, and each has it's own pitfalls. Lore codifies a solution that “just works” with lore install, which exactly replicates what will be run in production.

**Python 2 & 3 compatibility**

- pip install lore works regardless of whether your base system python is 2 or 3. Lore projects will always use the version of python specified in their runtime.txt
- Lore projects use the system service manager (upstart on ubuntu) instead of supervisord which requires python 2.

**Heroku_ buildpack compatibility CircleCI_, Domino_ , isc)**

- Lore supports runtime.txt to install and use a consistent version of python 2 or 3 in both development and production.
- lore install automatically manages freezing requirements.txt, using a virtualenv, so pip dependencies are exactly the same in development and production. This includes workarounds to support correctly (not) freezing github packages in requirements.txt

**Environment Specific Configuration**

- Lore supports reading environment variables from .env, for easy per project configuration. We recommend .gitignore .env and checking in a .env.template for developer reference to prevent leaking secrets.
- :python:`logging.getLogger(__name__)` is setup appropriately to console, file and/or syslog depending on environment
- syslog is replicated with structured data to loggly_ in production
- lore.util.timer logs info in development, and records  to librato_ in production
- Exception handling logs stack traces in development and test, but reports to rollbar_ in production
- lore console interactive python shell is color coded to prevent environmental confusion

**Multiple concurrent project compatibility**

- Lore manages a distinct python virtualenv for each project, which can be installed from scratch in development with lore install

**ISC compatibility**

- The commonly used virtualenvwrapper (and conda) breaks system python utilities, like isc, whenever you're working on a project. Lore works around this by bootstrapping into the appropriate virtualenv only when it is invoked by the developer.

**Binary library installation for MAXIMUM SPEED**

- Lore can build *tensorflow* from source when it is listed in requirements for development machines, which results in a 2-3x runtime training performance increase. Use lore install --native
- Lore also compiles *xgboost* on OS X with gcc-5 instead of clang to enable automatic parallelization

Lore Library
------------

**IO**

- :python:`lore.io.connection.Connection.select()` and :python:`Connection.dataframe()` can be automatically LRU cached to disk
- :python:`Connection` supports python %(name)s variable replacement in SQL
- :python:`Connection` statements are always annotated with metadata for pgHero
- :python:`Connection` is lazy, for fast startup, and avoids bootup errors in development with low connectivity
- :python:`Connection` supports multiple concurrent database connections

**Serialization**

- Lore serializers provide environment aware S3 distribution for keras/xgboost/scikit models
- Coming soon: heroku buildpack support for serialized models to marry the appropriate code for repeatable and deploys that can be safely rolled back

**Caching**

- Lore provides mulitple configurable cache types, RAM, Disk, coming soon: MemCached & Redis
- Disk cache is tested with pandas to avoid pitfalls encountered serializing w/ csv, h5py, pickle

**Encoders**

- Unique
- Discrete
- Quantile
- Norm

**Transformers**

- AreaCode
- EmailDomain
- NameAge
- NameSex
- NamePopulation
- NameFamilial

**Base Models**

- Abstract base classes for keras, xgboost, and scikit
  - inheriting class to define data(), encoders(), output_encoder(), benchmark()
  - multiple inheritance from custom base class w/ specific ABC for library
- provides hyper parameter optimization

**Fitting**

- Each call to Model.fit() saves the resulting model, along with the params to fit, epoch checkpoints and the resulting statistics, that can be reloaded, or uploaded with a Serializer

**Keras/Tensorflow**

- tensorboard support out of the box with tensorboard --logdir=models
- lore cleans up tensorflow before process exit to prevent spurious exceptions
- lore serializes Keras 2.0 models with extra care, to avoid several bugs (some that only appear at scale)
- ReloadBest callback early stops training on val_loss increase, and reloads the best epoch

**Utils**

- :python:`lore.util.timer` context manager writes to the log in development or librato in production*
- :python:`lore.util.timed` is a decorator for recording function execution wall time

Commands
--------

.. code-block:: bash

  $ lore api  #  start an api process
  $ lore console
  $ lore fit MODEL  #  train the model
  $ lore generate [all, api, model, notebook, task] NAME
  $ lore init [project]  #  create file structure
  $ lore install  #  setup dependencies in virtualenv
  $ lore test  #  make sure the project is in working order
  $ lore pip  #  launch pip in your virtual env
  $ lore python  # launch python in your virtual env
  $ lore notebook  # launch jupyter notebook in your virtual env


.. |circleci_badge| image:: https://circleci.com/gh/instacart/lore.png?style=shield&circle-token=54008e55ae13a0fa354203d13e7874c5efcb19a2
.. _Heroku: https://heroku.com/
.. _CircleCI: https://circleci.com/
.. _Domino: https://www.dominodatalab.com/
.. _loggly: https://www.loggly.com/
.. _librato: https://www.librato.com/
.. _rollbar: https://rollbar.com/
