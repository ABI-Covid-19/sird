from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Model:
    """
    SIRD model of Covid-19
    """

    NZ_POPULATION = 5000000
    NB_OF_STEPS = 100
    DELTA_T = 1 / NB_OF_STEPS

    class Parameter:
        """
        A model parameter, i.e. either a VOI or a state variable.
        """

        class Kind(Enum):
            VOI = auto()
            STATE = auto()

        def __init__(self, kind, name, initial_value):
            # Initialise our model parameter.

            self.__kind = kind
            self.__name = name
            self.__values = np.array([initial_value])

        def kind(self):
            # Return our kind.

            return self.__kind

        def name(self):
            # Return our name.

            return self.__name

        def values(self):
            # Return our values.

            return self.__values

        def __append_value(self, value):
            # Append the given value to our internal values.

            self.__values = np.append(self.__values, value)

    def __init__(self):
        # Initialise (i.e. reset) our SIRD model.

        self.reset()

    def reset(self):
        # Reset our SIRD model.

        self.__s = 997
        self.__i = 3
        self.__r = 0
        self.__d = 0
        self.__n = 1000
        self.__beta = 0.4
        self.__gamma = 0.035
        self.__mu = 0.005

        # Keep track of various model parameters.

        self.__voi_parameter = self.Parameter(self.Parameter.Kind.VOI, 'time', 0)
        self.__s_parameter = self.Parameter(self.Parameter.Kind.STATE, 'S', self.__s)
        self.__i_parameter = self.Parameter(self.Parameter.Kind.STATE, 'I', self.__i)
        self.__r_parameter = self.Parameter(self.Parameter.Kind.STATE, 'R', self.__r)
        self.__d_parameter = self.Parameter(self.Parameter.Kind.STATE, 'D', self.__d)

        self.parameters = {
            self.__voi_parameter.name(): self.__voi_parameter,
            self.__s_parameter.name(): self.__s_parameter,
            self.__i_parameter.name(): self.__i_parameter,
            self.__r_parameter.name(): self.__r_parameter,
            self.__d_parameter.name(): self.__d_parameter,
        }

    def run(self, sim_duration=100):
        # Make sure that we were given a valid simulation duration.

        if not isinstance(sim_duration, int) or sim_duration <= 0:
            print('The simulation duration (', sim_duration, ') must be an integer value greater than 0 (days).',
                  sep='')

            return

        # Run our SIRD simulation.

        for i in range(sim_duration):
            # Compute the SIRD model for one day.

            for j in range(Model.NB_OF_STEPS):
                self.__s -= Model.DELTA_T * (self.__beta * self.__i * self.__s / self.__n)
                self.__i += Model.DELTA_T * (
                        self.__beta * self.__i * self.__s / self.__n - self.__gamma * self.__i - self.__mu * self.__i)
                self.__r += Model.DELTA_T * (self.__gamma * self.__i)
                self.__d += Model.DELTA_T * (self.__mu * self.__i)

            # Update our simulation results.

            self.__voi_parameter._Parameter__append_value(i + 1)
            self.__s_parameter._Parameter__append_value(self.__s)
            self.__i_parameter._Parameter__append_value(self.__i)
            self.__r_parameter._Parameter__append_value(self.__r)
            self.__d_parameter._Parameter__append_value(self.__d)

    def plot(self):
        # Plot the results.

        plt.clf()  # In case there is already a Matplotlib window.
        plt.gcf().canvas.set_window_title('SIRD model')

        plt.plot(self.__voi_parameter.values(), self.__s_parameter.values(), '#0071bd', label=self.__s_parameter.name())
        plt.plot(self.__voi_parameter.values(), self.__i_parameter.values(), '#d9521a', label=self.__i_parameter.name())
        plt.plot(self.__voi_parameter.values(), self.__r_parameter.values(), '#edb020', label=self.__r_parameter.name())
        plt.plot(self.__voi_parameter.values(), self.__d_parameter.values(), '#7e2f8e', label=self.__d_parameter.name())
        plt.legend(loc='best')
        plt.xlabel('time (day)')

        plt.show()


if __name__ == '__main__':
    m = Model()

    m.run()
    m.plot()