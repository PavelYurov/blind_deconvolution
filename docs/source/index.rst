BlindDeconvolution Documentation
=================================

Фреймворк для исследования методов слепой деконволюции изображений.

.. toctree::
   :maxdepth: 3
   :caption: Руководство пользователя

   installation
   usage_guide
   configuration
   data_flow

.. toctree::
   :maxdepth: 3
   :caption: Архитектура и API

   architecture
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Теория методов

   theory/placeholder

.. toctree::
   :maxdepth: 2
   :caption: Разработка

   CONTRIBUTING
   CHANGELOG

.. toctree::
   :maxdepth: 4
   :caption: Автодокументация модулей

   modules


Быстрый старт
--------------

.. code-block:: python

   from blinddeconv.processing import Processing

   proc = Processing(images_folder="images/original", color=False)
   proc.read_all()
   proc.show()

Индексы
-------

* :ref:`modindex` — Индекс модулей
* :ref:`genindex` — Алфавитный индекс
* :ref:`search` — Поиск
