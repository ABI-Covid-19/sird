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

 # Create a default instance of the SIRD model, i.e. use all of the Covid-19
 # data for New Zealand. Optional parameters:
 #  - use: Model.Use.DATA (use the Covid-19 data from https://bit.ly/2X9zdos) or
 #         Model.Use.WIKIPEDIA (model parameter values from Wikipedia; see
 #         https://bit.ly/2VMvb6h), default: Use.DATA;
 #  - country: the country for which we want to use the Covid-19 data, default:
 #             'New Zealand'; and
 #  - max_data: the number n of data to use with n <= 0 meaning that all the
 #              Covid-19 data is used, default: 0.
 m = sird.Model()

 # Run the model for the default amount of days, i.e. 100 days. Optional
 # parameter:
 #  - nb_of_days: number of days n worth of simulation with n > 0, default: 100.
 m.run()

 # Plot the results. Optional parameters:
 #  - figure: an existing figure to which we want the plot to be added, default:
 #            None; and
 #  - two_axes: use a second axis for the data, default: False.
 m.plot()

 # Reset the model, re-run it for 150 days and plot its results.
 m.reset()
 m.run(150)
 m.plot()

 # Generate a movie showing the evolution of the SIRD model throughout time.
 # Note: this requires FFmpeg to be installed.
 m.movie('movie.mp4')

 # Output all the values for S, I, R and D.
 print(m.s())
 print(m.i())
 print(m.r())
 print(m.d())

 # Output the value for S, I, R and D at day 29.
 print(m.s(29))
 print(m.i(29))
 print(m.r(29))
 print(m.d(29))

For the first run, you should get something like:

.. image:: res/figure.png

As for the movie, you should get something like:

.. image:: res/movie.gif
