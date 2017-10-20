======
 Lore
======

|circleci_badge|


Lore is a python data science framework to design, fit, and exploit data science models from development to production. It codifies best practices to simplify collaborating and deploying models developed on a laptop with Jupyter notebook, into high availability distributed production data centers.


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



Update config/database.cfg to specify your database url:: ini

  # config/database.cfg\

  [MAIN]
  url: $DATABASE_URL

you can set environment variable for only the lore process with the .env file:: bash

  # .env

  DATABASE_URL=postgres://user:password@localhost:5432/main_development


Create a sql file that specifies your data

# app/extracts/subscribers.sql:: sql

  SELECT
    first_name,
    has_subscription
  FROM users
  LIMIT = %(limit)s

Pipelines are the unsexy, but essential component of most machine learning applications. They transform raw data into encoded training (and prediction) data for a model. Lore has several features to make data munging more palatable.:: python

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


The superclass `lore.pipelines.TrainTestSplit` will take care of:
1) splitting the data into training_data/validation_data/test_data dataframes
2) fitting the encoders to training_data
3) transforming training_data/validation_data/test_data for the model

Define some models that will fit and predict the data. Base models are designed to be extended and overridden, but work with defaults out of the box.:: python

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


Test the models predictive power:: python

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

Run tests:: bash

  $ lore test


Experiment and tune `notebooks/` with `$ lore notebook` using the app kernel





Python Modules
==============
Lore provides python modules to simplify and standardize Machine Learning techniques across multiple libraries.

Core Functionality
------------------
* **Models** are compatibility wrappers for your favorite library (keras, xgboost, scikit). They come with reasonable defaults for rough draft training out of the box.
* **Pipelines** fetch, encode, and split data into training/test sets for models. A single pipeline will have one Encoder per feature in the model.
* **Encoders** operate within Pipelines to transform a single feature into an optimal representation for learning.
* **Transformers** provide common operations, like extracting the area code from a free text phone number. They can be chained together inside encoders. They efficiently

Supporting functionality
------------------------
* **io.Connection** allows querying data from postgresql/redshift.
* **Serializers** persist models with their pipelines and encoders to disk, or s3.
* **Caches** save intermediate data, sometimes for reproducibility, sometimes for performance.

Utilities
---------
* **Logger** writes per environment (development/test/production) to ./logs/ + console if present + syslog in production
* **Timer** writes to the log in development or librato in production*


On the shoulders giants
===================================

Lore is designed to be as fast and powerful as the underlying libraries.
It seamlessly supports workflows that utilize:

* Keras/Tensorflow + Tensorboard
* XGBoost
* Scikit-Learn
* Jupyter Notebook
* Pandas
* Numpy
* Matplotlib
* SQLAlchemy, psychopg
* virtualenv, pyenv, python (2.7, 3.3+)


Commands
========

$ lore init
$ lore install [--upgade]
$ lore generate [**all**, api, model, notebook, task] NAME
$ lore test
$ lore console
$ lore api


Project Structure
=================

::

├── .env.template            <- Template for environment variables for developers (mirrors production)
├── runtime.txt              <- keeps dev and production in sync (pyenv or buildpack)
├── README.rst               <- The top-level README for developers using this project.
├── requirements.txt         <- keeps dev and production in sync via pip, managed w/ lore install
│
├── data/                    <- caches and intermediate data from pipelines
│
├── docs/                    <- generated from src
│
├── notebooks/               <- explorations of data and models
│       └── my_exploration/
│            └── exploration_1.ipynb
│
├── logs/
│   ├── development.log
│   └── test.log
│
├── models/                  <- persisted trained models
│
├── my_project/
│   ├── __init__.py          <- environment, logging, exceptions, metrics initializers
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── my_model.py      <- external entry points to my_model
│   │
│   ├── extracts/
│   │   └── my_model.sql     <- a query to fetch features for my_model's pipeline
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── my_model.py      <- define the pipeline for my_model
│   │
│   └── models/
│       ├── __init__.py
│       └── my_model.py      <- inherits and customized multiple lore base models
│
└── tests/
    ├── data/                <- test caches and intermediate data from pipelines
    ├── models/              <- persisted models for tests
    ├── mocks/               <- mock code to stub out models/pipelines etc
    └── unit/                <- unit tests for my_model



Design Philosophies & Inspiration
=================================

* Personal Pain
* Minimal Abstraction
* No code is better than no code (https://blog.codinghorror.com/the-best-code-is-no-code-at-all/)
* Convention over configuration (https://xkcd.com/927/)
* Sharp Tools (https://www.schneems.com/2016/08/16/sharp-tools.html)
* Rails (https://en.wikipedia.org/wiki/Ruby_on_Rails)
* Cookie Cutter Data Science (https://drivendata.github.io/cookiecutter-data-science/)
* Gene Roddenberry (https://www.youtube.com/watch?v=0JLgywxeaLM)

.. |circleci_badge| image:: https://circleci.com/gh/instacart/lore.png?style=shield&circle-token=54008e55ae13a0fa354203d13e7874c5efcb19a2