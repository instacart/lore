{NAME}
==========

System Setup
------------

1) Get your system setup

.. code::

  $ lore install

2) Set correct variables in `.env`

.. code::

  $ cp .env.template .env
  $ edit .env

Running
-------

The service runs on {DOMAIN}, to run locally:

.. code::

  $ lore console
  $ lore api

Testing
-------

CircleCI_ runs on push, to run locally:

.. code::

  $ lore test

Training
--------

Nightly training runs are scheduled on Domino_, to train locally:

.. code::

  $ lore notebook

Deploying
---------

.. code::

  $ isc launch {DOMAIN} {NAME}@master --follow

.. _CircleCI: https://circleci.com/
.. _Domino: https://domino.io/
