{{app_name}}
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

To run locally:

.. code::

  $ lore server

Testing
-------

To test locally:

.. code::

  $ lore test

Training
--------

To train locally:

.. code::

  $ lore fit MODEL

Deploying
---------

.. code::

  $ git push heroku master
