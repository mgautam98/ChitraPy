Welcome to ChitraPy's documentation!
====================================

ChitraPy is a digital Image Processing Library in Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
==================
Install from Pypi

.. code-block:: bash
   :linenos:

   pip install ChitraPy



Install from source

.. code-block:: bash
   :linenos:

   git clone https://github.com/mgautam98/ChitraPy.git  
   cd ChitraPy  
   python3 setup.py install

Usage
==================

.. code-block:: python
   :linenos:
   :caption: sample.py
   :name: sample-py

   from ChitraPy import filters, helpers
   import matplotlib.pyplot as plt

   # Load a sample Image

   !wget https://i.imgur.com/D24n5DL.png
   img = plt.imread('./D24n5DL.png')
   plt.imshow(img)

   # invert an image
   invert = filters.invert(img)
   plt.imshow(invert)


Filters
==================
.. automodule:: ChitraPy.filters
   :members:

Helpers
==================
.. automodule:: ChitraPy.helpers
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
