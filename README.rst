SIRD
====

A Python script to model Covid-19 using the `SIRD model <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIRD_model>`_.

Install/upgrade
---------------

::

 pip install -U git+https://github.com/ABI-Covid-19/sird.git

Uninstall
---------

::

 pip uninstall -y sird

Use
---

::

 # Import the SIRD module.
 import sird

 # Create an instance of the SIRD model.
 m = sird.Model()

 # Run the model for 100 days (default) and plot its results.
 m.run()
 m.plot()

 # Reset the model, re-run it for 150 days and plot its results.
 m.reset()
 m.run(150)
 m.plot()

 # Output the kind, name and values of the model's time parameter.
 t = m.parameters['time']
 print(' [', t.kind(), '] ', t.name(), ': ', t.values(), sep='')

 # Output the kind, name and values of all the model's parameters.
 for p in m.parameters.values():
     print(' [', p.kind(), '] ', p.name(), ': ', p.values(), sep='')

For the first run, you should get something like:

.. image:: res/figure.png
