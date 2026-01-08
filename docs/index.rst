BlindDeconvolution Documentation
=================================

Фреймворк для исследования методов слепой деконволюции.

.. toctree::
   :maxdepth: 2
   :caption: Содержание:

   modules

Быстрый старт
-------------

.. code-block:: python

   from processing import Processing
   from algorithms.implementations.richardson_lucy import RichardsonLucy

   proc = Processing()
   proc.read_all()
   proc.process(RichardsonLucy({'iter': 50}))
   proc.show()

Модули
------

* :ref:`modindex` — Индекс модулей
* :ref:`genindex` — Алфавитный индекс
* :ref:`search` — Поиск

