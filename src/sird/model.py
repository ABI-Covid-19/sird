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

    __moh_data = None

    def __init__(self, use_moh_data=True):
        # Retrieve some MoH data, if needed.

        if use_moh_data and Model.__moh_data == None:
            Model.__moh_data = pd.read_csv(
                'https://docs.google.com/spreadsheets/u/1/d/16UMnHbnBHju-fK45aSdaJhVmrXJpy71oxSiN_AvqV84/export?format=csv&id=16UMnHbnBHju-fK45aSdaJhVmrXJpy71oxSiN_AvqV84&gid=0',
                usecols=[7, 9, 11])

        # Keep track of whether to use the MoH data.

        self.__use_moh_data = use_moh_data

        # Initialise (i.e. reset) our SIRD model.

        self.reset()

    def __moh_s(self, day):
        return Model.NZ_POPULATION - self.__moh_i(day) - self.__moh_r(day) - self.__moh_d(day)

    def __moh_i(self, day):
        return Model.__moh_data.iloc[day][2]

    def __moh_r(self, day):
        return Model.__moh_data.iloc[day][0]

    def __moh_d(self, day):
        return Model.__moh_data.iloc[day][1]

    def reset(self):
        # Reset our SIRD model.

        if self.__use_moh_data:
            self.__s = self.__moh_s(0)
            self.__i = self.__moh_i(0)
            self.__r = self.__moh_r(0)
            self.__d = self.__moh_d(0)
            self.__n = Model.NZ_POPULATION
        else:
            self.__s = 997
            self.__i = 3
            self.__r = 0
            self.__d = 0
            self.__n = 1000

        self.__beta = 0.4
        self.__gamma = 0.035
        self.__mu = 0.005

        # Reset our simulation values.

        self.__s_values = np.array([self.__s])
        self.__i_values = np.array([self.__i])
        self.__r_values = np.array([self.__r])
        self.__d_values = np.array([self.__d])

    def run(self, sim_duration=100):
        # Make sure that we were given a valid simulation duration.

        if not isinstance(sim_duration, int) or sim_duration <= 0:
            print('The simulation duration (', sim_duration, ') must be an integer value greater than 0 (days).',
                  sep='')

            return

        # Run our SIRD simulation.

        for i in range(sim_duration):
            # Output the MoH data for the given day, if needed.

            if self.__use_moh_data:
                try:
                    print('Day ', i, ': S=', self.__moh_s(i), ' I=', self.__moh_i(i), ' R=', self.__moh_r(i), ' D=',
                          self.__moh_d(i), sep='')
                except:
                    pass

            # Compute the SIRD model for one day.

            for j in range(Model.NB_OF_STEPS):
                self.__s -= Model.DELTA_T * (self.__beta * self.__i * self.__s / self.__n)
                self.__i += Model.DELTA_T * (
                        self.__beta * self.__i * self.__s / self.__n - self.__gamma * self.__i - self.__mu * self.__i)
                self.__r += Model.DELTA_T * (self.__gamma * self.__i)
                self.__d += Model.DELTA_T * (self.__mu * self.__i)

            # Update our simulation results.

            self.__s_values = np.append(self.__s_values, self.__s)
            self.__i_values = np.append(self.__i_values, self.__i)
            self.__r_values = np.append(self.__r_values, self.__r)
            self.__d_values = np.append(self.__d_values, self.__d)

    def plot(self):
        # Plot the results.

        plt.clf()  # In case there is already a Matplotlib window.
        plt.gcf().canvas.set_window_title('SIRD model')

        days = range(self.__s_values.size)

        plt.plot(days, self.__s_values, '#0071bd', label='S')
        plt.plot(days, self.__i_values, '#d9521a', label='I')
        plt.plot(days, self.__r_values, '#edb020', label='R')
        plt.plot(days, self.__d_values, '#7e2f8e', label='D')
        plt.legend(loc='best')
        plt.xlabel('time (day)')

        plt.show()

    def s(self, day=-1):
        if day == -1:
            return self.__s_values
        else:
            return self.__s_values[day]

    def i(self, day=-1):
        if day == -1:
            return self.__i_values
        else:
            return self.__i_values[day]

    def r(self, day=-1):
        if day == -1:
            return self.__r_values
        else:
            return self.__r_values[day]

    def d(self, day=-1):
        if day == -1:
            return self.__d_values
        else:
            return self.__d_values[day]


if __name__ == '__main__':
    m = Model()

    m.run()
    m.plot()
