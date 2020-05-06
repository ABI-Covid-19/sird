import math
from enum import Enum, auto

import matplotlib
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter


class Model:
    """
    SIRD model of Covid-19.
    """

    __NZ_POPULATION = 5000000
    __CONFIRMED_URL = 'https://bit.ly/35yJO0d'
    __RECOVERED_URL = 'https://bit.ly/2L6jLE9'
    __DEATHS_URL = 'https://bit.ly/2L0hzxQ'
    __N_FILTERED = 6  # Number of state variables to filter (I, R, D, β, γ and μ).
    __N_MEASURED = 3  # Number of measured variables (I, R and D).
    __NB_OF_STEPS = 100
    __DELTA_T = 1 / __NB_OF_STEPS
    __FIG_SIZE = (11, 13)
    __S_COLOR = '#0072bd'
    __I_COLOR = '#d95319'
    __R_COLOR = '#edb120'
    __D_COLOR = '#7e2f8e'
    __BETA_COLOR = '#77ac30'
    __GAMMA_COLOR = '#4dbeee'
    __MU_COLOR = '#a2142f'
    __MOH_DATA_ALPHA = 0.3
    __MOH_DATA = None

    class Use(Enum):
        WIKIPEDIA = auto()
        MOH_DATA = auto()

    def __init__(self, use=Use.MOH_DATA, max_data=-1):
        """
        Initialise our Model object.
        """

        # Retrieve the MoH data (if requested and needed).

        if use == Model.Use.MOH_DATA and Model.__MOH_DATA is None:
            confirmed_data = self.__jhu_data(Model.__CONFIRMED_URL)
            recovered_data = self.__jhu_data(Model.__RECOVERED_URL)
            deaths_data = self.__jhu_data(Model.__DEATHS_URL)

            for i in range(confirmed_data.shape[1]):
                c = confirmed_data.iloc[0][i]
                r = recovered_data.iloc[0][i]
                d = deaths_data.iloc[0][i]
                data = [c - r - d, r, d]

                if Model.__MOH_DATA is None:
                    Model.__MOH_DATA = np.array(data)
                else:
                    Model.__MOH_DATA = np.vstack((Model.__MOH_DATA, data))

        if use == Model.Use.MOH_DATA:
            self.__moh_data = Model.__MOH_DATA
        else:
            self.__moh_data = None

        if self.__moh_data is not None and max_data != -1:
            self.__moh_data = self.__moh_data[:max_data]

        # Keep track of whether to use the MoH data.

        self.__use_moh_data = use == Model.Use.MOH_DATA

        # Declare some internal variables (that will then be initialised through our call to reset()).

        self.__beta = None
        self.__gamma = None
        self.__mu = None

        self.__ukf = None

        self.__x = None
        self.__n = None

        self.__moh_data_s_values = None
        self.__moh_data_i_values = None
        self.__moh_data_r_values = None
        self.__moh_data_d_values = None

        self.__s_values = None
        self.__i_values = None
        self.__r_values = None
        self.__d_values = None

        self.__beta_values = None
        self.__gamma_values = None
        self.__mu_values = None

        # Initialise (i.e. reset) our SIRD model.

        self.reset()

    @staticmethod
    def __jhu_data(url):
        data = pd.read_csv(url)
        data = data[data['Country/Region'] == 'New Zealand']
        data = data.drop(data.columns[list(range(41))], axis=1)

        return data

    def __moh_data_x(self, day, index):
        """
        Return the MoH I/R/D value for the given day.
        """

        return self.__moh_data[day][index] if self.__use_moh_data else math.nan

    def __moh_data_s(self, day):
        """
        Return the MoH S value for the given day.
        """

        if self.__use_moh_data:
            return Model.__NZ_POPULATION - self.__moh_data_i(day) - self.__moh_data_r(day) - self.__moh_data_d(day)
        else:
            return math.nan

    def __moh_data_i(self, day):
        """
        Return the MoH I value for the given day.
        """

        return self.__moh_data_x(day, 0)

    def __moh_data_r(self, day):
        """
        Return the MoH R value for the given day.
        """

        return self.__moh_data_x(day, 1)

    def __moh_data_d(self, day):
        """
        Return the MoH D value for the given day.
        """

        return self.__moh_data_x(day, 2)

    def __moh_data_available(self, day):
        """
        Return whether some data is available for the given day.
        """

        return day <= self.__moh_data.shape[0] - 1 if self.__use_moh_data else False

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

    @staticmethod
    def __f(x, dt, **kwargs):
        """
        State function.

        The ODE system to solve is:
          dI/dt = βIS/N - γI - μI
          dR/dt = γI
          dD/dt = μI
        """

        model_self = kwargs.get('model_self')
        with_ukf = kwargs.get('with_ukf', True)

        if with_ukf:
            s = model_self.__n - x[:3].sum()
            beta = x[3]
            gamma = x[4]
            mu = x[5]
        else:
            s = model_self.__n - x.sum()
            beta = model_self.__beta
            gamma = model_self.__gamma
            mu = model_self.__mu

        a = np.array([[1 + dt * (beta * s / model_self.__n - gamma - mu), 0, 0, 0, 0, 0],
                      [dt * gamma, 1, 0, 0, 0, 0],
                      [dt * mu, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        if with_ukf:
            return a @ x
        else:
            return a[:3, :3] @ x

    @staticmethod
    def __h(x):
        """
        Measurement function.
        """

        return x[:Model.__N_MEASURED]

    def reset(self):
        """
        Reset our SIRD model.
        """

        # Reset β, γ and μ to the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        self.__beta = 0.4
        self.__gamma = 0.035
        self.__mu = 0.005

        # Reset I, R and D to the MoH data at day 0 or the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        if self.__use_moh_data:
            self.__x = np.array([self.__moh_data_i(0), self.__moh_data_r(0), self.__moh_data_d(0)])
            self.__n = Model.__NZ_POPULATION
        else:
            self.__x = np.array([3, 0, 0])
            self.__n = 1000

        # Reset our Unscented Kalman filter (if required). Note tat we use a dt value of 1 (day) and not the value of
        # Model.__DELTA_T.

        if self.__use_moh_data:
            points = MerweScaledSigmaPoints(Model.__N_FILTERED,
                                            1e-3,  # Alpha value (usually a small positive value like 1e-3).
                                            2,  # Beta value (a value of 2 is optimal for a Gaussian distribution).
                                            0,  # Kappa value (usually, either 0 or 3-n).
                                            )

            self.__ukf = UnscentedKalmanFilter(Model.__N_FILTERED, Model.__N_MEASURED, 1, self.__h, Model.__f, points)

            self.__ukf.x = np.array(
                [self.__moh_data_i(0), self.__moh_data_r(0), self.__moh_data_d(0), self.__beta, self.__gamma, self.__mu])

        # Reset our MoH data (if requested).

        if self.__use_moh_data:
            self.__moh_data_s_values = np.array([self.__moh_data_s(0)])
            self.__moh_data_i_values = np.array([self.__moh_data_i(0)])
            self.__moh_data_r_values = np.array([self.__moh_data_r(0)])
            self.__moh_data_d_values = np.array([self.__moh_data_d(0)])

        # Reset our predicted/estimated values.

        self.__s_values = np.array([self.__s_value()])
        self.__i_values = np.array([self.__i_value()])
        self.__r_values = np.array([self.__r_value()])
        self.__d_values = np.array([self.__d_value()])

        # Reset our estimated SIRD model parameters.

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
            # Compute our predicted/estimated state by computing our SIRD model / Unscented Kalman filter for one day.

            if self.__use_moh_data and self.__moh_data_available(i):
                self.__ukf.predict(model_self=self)
                self.__ukf.update(np.array([self.__moh_data_i(i), self.__moh_data_r(i), self.__moh_data_d(i)]))

                self.__x = self.__ukf.x[:3]
                self.__beta = self.__ukf.x[3]
                self.__gamma = self.__ukf.x[4]
                self.__mu = self.__ukf.x[5]
            else:
                for j in range(1, Model.__NB_OF_STEPS + 1):
                    self.__x = Model.__f(self.__x, Model.__DELTA_T, model_self=self, with_ukf=False)

            # Keep track of our MoH data (if requested).

            if self.__use_moh_data:
                if self.__moh_data_available(i):
                    self.__moh_data_s_values = np.append(self.__moh_data_s_values, self.__moh_data_s(i))
                    self.__moh_data_i_values = np.append(self.__moh_data_i_values, self.__moh_data_i(i))
                    self.__moh_data_r_values = np.append(self.__moh_data_r_values, self.__moh_data_r(i))
                    self.__moh_data_d_values = np.append(self.__moh_data_d_values, self.__moh_data_d(i))
                else:
                    self.__moh_data_s_values = np.append(self.__moh_data_s_values, math.nan)
                    self.__moh_data_i_values = np.append(self.__moh_data_i_values, math.nan)
                    self.__moh_data_r_values = np.append(self.__moh_data_r_values, math.nan)
                    self.__moh_data_d_values = np.append(self.__moh_data_d_values, math.nan)

            # Keep track of our predicted/estimated values.

            self.__s_values = np.append(self.__s_values, self.__s_value())
            self.__i_values = np.append(self.__i_values, self.__i_value())
            self.__r_values = np.append(self.__r_values, self.__r_value())
            self.__d_values = np.append(self.__d_values, self.__d_value())

            # Keep track of our estimated SIRD model parameters.

            self.__beta_values = np.append(self.__beta_values, self.__beta)
            self.__gamma_values = np.append(self.__gamma_values, self.__gamma)
            self.__mu_values = np.append(self.__mu_values, self.__mu)

    def plot(self, fig=None, two_axes=False):
        """
        Plot the results using five subplots for 1) S, 2) I and R, 3) D, 4) β, and 5) γ and μ. In each subplot, we plot
        the MoH data (if requested) as bars and the computed value as a line.
        """

        days = range(self.__s_values.size)
        nrows = 5 if self.__use_moh_data else 3
        ncols = 1

        if fig is None:
            show_fig = True
            fig, ax = plt.subplots(nrows, ncols, figsize=Model.__FIG_SIZE)
        else:
            fig.clf()

            show_fig = False
            ax = fig.subplots(nrows, ncols)

        fig.canvas.set_window_title('SIRD model fitted to MoH data' if self.__use_moh_data else 'Wikipedia SIRD model')

        # First subplot: S.

        ax1 = ax[0]
        ax1.plot(days, self.__s_values, Model.__S_COLOR, label='S')
        ax1.legend(loc='best')
        if self.__use_moh_data:
            ax2 = ax1.twinx() if two_axes else ax1
            ax2.bar(days, self.__moh_data_s_values, color=Model.__S_COLOR, alpha=Model.__MOH_DATA_ALPHA)
            data_s_range = Model.__NZ_POPULATION - min(self.__moh_data_s_values)
            data_block = 10 ** (math.floor(math.log10(data_s_range)) - 1)
            s_values_shift = data_block * math.ceil(data_s_range / data_block)
            ax2.set_ylim(Model.__NZ_POPULATION - s_values_shift, Model.__NZ_POPULATION)

        # Second subplot: I and R.

        ax1 = ax[1]
        ax1.plot(days, self.__i_values, Model.__I_COLOR, label='I')
        ax1.plot(days, self.__r_values, Model.__R_COLOR, label='R')
        ax1.legend(loc='best')
        if self.__use_moh_data:
            ax2 = ax1.twinx() if two_axes else ax1
            ax2.bar(days, self.__moh_data_i_values, color=Model.__I_COLOR, alpha=Model.__MOH_DATA_ALPHA)
            ax2.bar(days, self.__moh_data_r_values, color=Model.__R_COLOR, alpha=Model.__MOH_DATA_ALPHA)

        # Third subplot: D.

        ax1 = ax[2]
        ax1.plot(days, self.__d_values, Model.__D_COLOR, label='D')
        ax1.legend(loc='best')
        if self.__use_moh_data:
            ax2 = ax1.twinx() if two_axes else ax1
            ax2.bar(days, self.__moh_data_d_values, color=Model.__D_COLOR, alpha=Model.__MOH_DATA_ALPHA)

        # Fourth subplot: β.

        if self.__use_moh_data:
            ax1 = ax[3]
            ax1.plot(days, self.__beta_values, Model.__BETA_COLOR, label='β')
            ax1.legend(loc='best')

        # Fourth subplot: γ and μ.

        if self.__use_moh_data:
            ax1 = ax[4]
            ax1.plot(days, self.__gamma_values, Model.__GAMMA_COLOR, label='γ')
            ax1.plot(days, self.__mu_values, Model.__MU_COLOR, label='μ')
            ax1.legend(loc='best')

        plt.xlabel('time (day)')

        if show_fig:
            plt.show()

    def movie(self, filename):
        if self.__use_moh_data:
            data_size = Model.__MOH_DATA.shape[0]
            fig = plt.figure(figsize=Model.__FIG_SIZE)
            backend = matplotlib.get_backend()
            writer = manimation.writers['ffmpeg'](15)

            matplotlib.use("Agg")

            with writer.saving(fig, filename, 96):
                for i in range(1, data_size + 1):
                    print('Processing frame #', i, '/', data_size, '...', sep='')

                    self.__moh_data = Model.__MOH_DATA[:i]

                    self.reset()
                    self.run()
                    self.plot(fig=fig)

                    writer.grab_frame()

                print('All done!')

            matplotlib.use(backend)

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
