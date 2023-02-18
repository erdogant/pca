.. include:: add_top.add

Installation
##############

Create environment
*********************


If desired, install ``pca`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_pca python=3.8
    conda activate env_pca


Pip install
*********************

.. code-block:: console

    # Install from Pypi:
    pip install pca

    # Install directly from github
    pip install git+https://github.com/erdogant/pca


Uninstalling
##############

Remove the environment with installation:

.. code-block:: console

   # List all the active environments. pca should be listed.
   conda env list

   # Remove the pca environment
   conda env remove --name pca

   # List all the active environments. pca should be absent.
   conda env list



.. include:: add_bottom.add