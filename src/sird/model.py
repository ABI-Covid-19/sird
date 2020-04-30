import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Model:
    """
    SIRD model of Covid-19.
    Note that N = S+I+R+D and that we have data from the MoH for I, R and D. So, there is no need to compute S as such.
    """

    NZ_POPULATION = 5000000
    NB_OF_STEPS = 100
    DELTA_T = 1 / NB_OF_STEPS
    S_COLOR = '#0071bd'
    I_COLOR = '#d9521a'
    R_COLOR = '#edb020'
    D_COLOR = '#7e2f8e'
    BETA_COLOR = '#7e2f8e'
    GAMMA_COLOR = '#4dbeee'
    MU_COLOR = '#a2142f'
    MOH_ALPHA = 0.3

    __moh_data = None

    def __init__(self, use_moh_data=True):
        """
        Initialise our Model object.
        """

        # Retrieve the MoH data (if requested).

        if use_moh_data and Model.__moh_data == None:
            Model.__moh_data = pd.read_csv(
                'https://docs.google.com/spreadsheets/u/1/d/16UMnHbnBHju-fK45aSdaJhVmrXJpy71oxSiN_AvqV84/export?format=csv&id=16UMnHbnBHju-fK45aSdaJhVmrXJpy71oxSiN_AvqV84&gid=0',
                usecols=[7, 9, 11])

        # Keep track of whether to use the MoH data.

        self.__use_moh_data = use_moh_data

        # Initialise (i.e. reset) our SIRD model.

        self.reset()

    def __moh_s(self, day):
        """
        Return the MoH S value for the given day.
        """

        return Model.NZ_POPULATION - self.__moh_i(day) - self.__moh_r(day) - self.__moh_d(day)

    def __moh_i(self, day):
        """
        Return the MoH I value for the given day.
        """

        return Model.__moh_data.iloc[day][2]

    def __moh_r(self, day):
        """
        Return the MoH R value for the given day.
        """

        return Model.__moh_data.iloc[day][0]

    def __moh_d(self, day):
        """
        Return the MoH D value for the given day.
        """

        return Model.__moh_data.iloc[day][1]

    def __moh_data_available(self, day):
        """
        Return whether some MoH data is available for the given day.
        """

        return day < Model.__moh_data.shape[0] if self.__use_moh_data else False

    def __s_value(self):
        """
        Return the S value based on the values of I, R, D and N.
        """

        return self.__n - self.__x.sum()

    def __i_value(self):
        """
        Return the I value.
        """

        return self.__x[0]

    def __r_value(self):
        """
        Return the R value.
        """

        return self.__x[1]

    def __d_value(self):
        """
        Return the D value.
        """

        return self.__x[2]

    def reset(self):
        """
        Reset our SIRD model.
        """

        if self.__use_moh_data:
            # We use the MoH data at day 0 as our initial guess for S, I, R and D.

            self.__x = np.array([self.__moh_i(0), self.__moh_r(0), self.__moh_d(0)])
            self.__n = Model.NZ_POPULATION
        else:
            # Use the (initial) values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

            self.__x = np.array([3, 0, 0])
            self.__n = 1000

        # Use the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        self.__beta = 0.4
        self.__gamma = 0.035
        self.__mu = 0.005

        # Reset our MoH data and simulation values.

        if self.__use_moh_data:
            self.__moh_s_values = np.array([self.__moh_s(0)])
            self.__moh_i_values = np.array([self.__moh_i(0)])
            self.__moh_r_values = np.array([self.__moh_r(0)])
            self.__moh_d_values = np.array([self.__moh_d(0)])

        self.__s_values = np.array([self.__s_value()])
        self.__i_values = np.array([self.__i_value()])
        self.__r_values = np.array([self.__r_value()])
        self.__d_values = np.array([self.__d_value()])

        self.__beta_values = np.array([self.__beta])
        self.__gamma_values = np.array([self.__gamma])
        self.__mu_values = np.array([self.__mu])

    def run(self, nb_of_days=100):
        """
        Run our SIRD model for the given number of days, taking advantage of the MoH data (if requested) to estimate the
        values of β, γ and μ.
        """

        # Make sure that we were given a valid number of days.

        if not isinstance(nb_of_days, int) or nb_of_days <= 0:
            print('The number of days (', nb_of_days, ') must be an integer value greater than 0.', sep='')

            return

        # Run our SIRD simulation.

        for i in range(1, nb_of_days + 1):
            # Compute the SIRD model for one day using:
            #   dI/dt = βIS/N - γI - μI
            #   dR/dt = γI
            #   dD/dt = μI

            for j in range(Model.NB_OF_STEPS):
                a = np.array(
                    [[1 + Model.DELTA_T * (self.__beta * self.__s_value() / self.__n - self.__gamma - self.__mu), 0, 0],
                     [Model.DELTA_T * self.__gamma, 1, 0],
                     [Model.DELTA_T * self.__mu, 0, 1]])
                self.__x = a.dot(self.__x)

            # Update our MoH data (if requested) and simulation values.

            if self.__moh_data_available(i):
                self.__moh_s_values = np.append(self.__moh_s_values, self.__moh_s(i))
                self.__moh_i_values = np.append(self.__moh_i_values, self.__moh_i(i))
                self.__moh_r_values = np.append(self.__moh_r_values, self.__moh_r(i))
                self.__moh_d_values = np.append(self.__moh_d_values, self.__moh_d(i))
            else:
                self.__moh_s_values = np.append(self.__moh_s_values, math.nan)
                self.__moh_i_values = np.append(self.__moh_i_values, math.nan)
                self.__moh_r_values = np.append(self.__moh_r_values, math.nan)
                self.__moh_d_values = np.append(self.__moh_d_values, math.nan)

            self.__s_values = np.append(self.__s_values, self.__s_value())
            self.__i_values = np.append(self.__i_values, self.__i_value())
            self.__r_values = np.append(self.__r_values, self.__r_value())
            self.__d_values = np.append(self.__d_values, self.__d_value())

            self.__beta_values = np.append(self.__beta_values, self.__beta)
            self.__gamma_values = np.append(self.__gamma_values, self.__gamma)
            self.__mu_values = np.append(self.__mu_values, self.__mu)

    def plot(self):
        """
        Plot the results using five subplots for 1) S, 2) I and R, 3) D, 4) β, and 5) γ and μ. In each subplot, we plot
        the MoH data (if requested) as bars and the computed value as a line.
        """

        days = range(self.__s_values.size)
        fig, ax = plt.subplots(5 if self.__use_moh_data else 3, 1)

        fig.canvas.set_window_title('SIRD model fitted to MoH data' if self.__use_moh_data else 'Default SIRD model')

        # First subplot: S.

        ax1 = ax[0]
        ax1.plot(days, self.__s_values, Model.S_COLOR, label='S')
        ax1.legend(loc='best')
        if self.__use_moh_data:
            ax2 = ax1.twinx()
            ax2.bar(days, self.__moh_s_values - min(self.__moh_s_values), color=Model.S_COLOR, alpha=Model.MOH_ALPHA,
                    label='MoH S')
            ax2.set_yticklabels(np.linspace(min(self.__moh_s_values), Model.NZ_POPULATION))

        # Second subplot: I and R.

        ax1 = ax[1]
        ax1.plot(days, self.__i_values, Model.I_COLOR, label='I')
        ax1.plot(days, self.__r_values, Model.R_COLOR, label='R')
        ax1.legend(loc='best')
        if self.__use_moh_data:
            ax2 = ax1.twinx()
            ax2.bar(days, self.__moh_i_values, color=Model.I_COLOR, alpha=Model.MOH_ALPHA, label='MoH I')
            ax2.bar(days, self.__moh_r_values, color=Model.R_COLOR, alpha=Model.MOH_ALPHA, label='MoH R')

        # Third subplot: D.

        ax1 = ax[2]
        ax1.plot(days, self.__d_values, Model.D_COLOR, label='D')
        ax1.legend(loc='best')
        if self.__use_moh_data:
            ax2 = ax1.twinx()
            ax2.bar(days, self.__moh_d_values, color=Model.D_COLOR, alpha=Model.MOH_ALPHA, label='MoH D')

        # Fourth subplot: β.

        if self.__use_moh_data:
            ax1 = ax[3]
            ax1.plot(days, self.__beta_values, Model.BETA_COLOR, label='β')
            ax1.legend(loc='best')

        # Fourth subplot: γ and μ.

        if self.__use_moh_data:
            ax1 = ax[4]
            ax1.plot(days, self.__gamma_values, Model.GAMMA_COLOR, label='γ')
            ax1.plot(days, self.__mu_values, Model.MU_COLOR, label='μ')
            ax1.legend(loc='best')

        plt.xlabel('time (day)')
        plt.show()

    def s(self, day=-1):
        """
        Return all the S values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__s_values
        else:
            return self.__s_values[day]

    def i(self, day=-1):
        """
        Return all the I values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__i_values
        else:
            return self.__i_values[day]

    def r(self, day=-1):
        """
        Return all the R values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__r_values
        else:
            return self.__r_values[day]

    def d(self, day=-1):
        """
        Return all the D values (if day=-1) or its value for a given day.
        """

        if day == -1:
            return self.__d_values
        else:
            return self.__d_values[day]


if __name__ == '__main__':
    # Create an instance of the SIRD model, asking for the MoH data to be used.

    m = Model()

    # Run the model and plot its S, I, R and D values, together with the MoH data.

    m.run()
    m.plot()
