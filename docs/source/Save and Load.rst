Saving
################

Saving and loading models is desired as the learning proces of a model for ``XXX`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`XXX.save` and function :func:`XXX.load`
Below we illustrate how to save and load models.


Saving a learned model can be done using the function :func:`XXX.save`:

.. code:: python

    import XXX

    # Load example data
    X,y_true = XXX.load_example()

    # Learn model
    model = XXX.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = XXX.save(model, 'learned_model_v1')



Loading
################


Loading a learned model can be done using the function :func:`XXX.load`:

.. code:: python

    import XXX

    # Load model
    model = XXX.load(model, 'learned_model_v1')

.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
