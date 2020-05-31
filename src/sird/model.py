import math
import sys
from enum import Enum, auto

import matplotlib
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter


class Model:
    """
    SIRD model of Covid-19.
    """

    __CONFIRMED_URL = 'https://bit.ly/35yJO0d'
    __RECOVERED_URL = 'https://bit.ly/2L6jLE9'
    __DEATHS_URL = 'https://bit.ly/2L0hzxQ'
    __POPULATION_URL = 'https://bit.ly/2WYjZCD'
    __JHU_DATA_SHIFT = 4
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
    __DATA_ALPHA = 0.3
    __DATA = None
    __POPULATION = None

    class Use(Enum):
        WIKIPEDIA = auto()
        DATA = auto()

    def __init__(self, use=Use.DATA, country='New Zealand', max_data=0):
        """
        Initialise our Model object.
        """

        # Retrieve the data (if requested and needed).

        if use == Model.Use.DATA and Model.__DATA is None:
            confirmed_data, confirmed_data_start = self.__jhu_data(Model.__CONFIRMED_URL, country)
            recovered_data, recovered_data_start = self.__jhu_data(Model.__RECOVERED_URL, country)
            deaths_data, deaths_data_start = self.__jhu_data(Model.__DEATHS_URL, country)
            data_start = min(confirmed_data_start, recovered_data_start, deaths_data_start) - Model.__JHU_DATA_SHIFT

            for i in range(data_start, confirmed_data.shape[1]):
                c = confirmed_data.iloc[0][i]
                r = recovered_data.iloc[0][i]
                d = deaths_data.iloc[0][i]
                data = [c - r - d, r, d]

                if Model.__DATA is None:
                    Model.__DATA = np.array(data)
                else:
                    Model.__DATA = np.vstack((Model.__DATA, data))

        if use == Model.Use.DATA:
            self.__data = Model.__DATA
        else:
            self.__data = None

        if self.__data is not None:
            if not isinstance(max_data, int):
                sys.exit('Error: \'max_data\' must be an integer value.')

            if max_data > 0:
                self.__data = self.__data[:max_data]

        # Retrieve the population (if needed).

        if Model.__POPULATION is None:
            Model.__POPULATION = {}

            response = requests.get(Model.__POPULATION_URL)
            soup = BeautifulSoup(response.text, 'html.parser')
            data = soup.select('div div div div div tbody tr')

            for i in range(len(data)):
                country_soup = BeautifulSoup(data[i].prettify(), 'html.parser')
                country_value = country_soup.select('tr td a')[0].get_text().strip()
                population_value = country_soup.select('tr td')[2].get_text().strip().replace(',', '')

                Model.__POPULATION[country_value] = int(population_value)

        if country in Model.__POPULATION:
            self.__population = Model.__POPULATION[country]
        else:
            sys.exit('Error: no population data is available for {}.'.format(country))

        # Keep track of whether to use the data.

        self.__use_data = use == Model.Use.DATA

        # Declare some internal variables (that will then be initialised through our call to reset()).

        self.__beta = None
        self.__gamma = None
        self.__mu = None

        self.__ukf = None

        self.__x = None
        self.__n = None

        self.__data_s_values = None
        self.__data_i_values = None
        self.__data_r_values = None
        self.__data_d_values = None

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
    def __jhu_data(url, country):
        data = pd.read_csv(url)
        data = data[(data['Country/Region'] == country) & data['Province/State'].isnull()]

        if data.shape[0] == 0:
            sys.exit('Error: no Covid-19 data is available for {}.'.format(country))

        data = data.drop(data.columns[list(range(Model.__JHU_DATA_SHIFT))], axis=1)  # Skip non-data columns.
        start = None

        for i in range(data.shape[1]):
            if data.iloc[0][i] != 0:
                start = Model.__JHU_DATA_SHIFT + i

                break

        return data, start

    def __data_x(self, day, index):
        """
        Return the I/R/D value for the given day.
        """

        return self.__data[day][index] if self.__use_data else math.nan

    def __data_s(self, day):
        """
        Return the S value for the given day.
        """

        if self.__use_data:
            return self.__population - self.__data_i(day) - self.__data_r(day) - self.__data_d(day)
        else:
            return math.nan

    def __data_i(self, day):
        """
        Return the I value for the given day.
        """

        return self.__data_x(day, 0)

    def __data_r(self, day):
        """
        Return the R value for the given day.
        """

        return self.__data_x(day, 1)

    def __data_d(self, day):
        """
        Return the D value for the given day.
        """

        return self.__data_x(day, 2)

    def __data_available(self, day):
        """
        Return whether some data is available for the given day.
        """

        return day <= self.__data.shape[0] - 1 if self.__use_data else False

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

        # Reset I, R and D to the data at day 0 or the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        if self.__use_data:
            self.__x = np.array([self.__data_i(0), self.__data_r(0), self.__data_d(0)])
            self.__n = self.__population
        else:
            self.__x = np.array([3, 0, 0])
            self.__n = 1000

        # Reset our Unscented Kalman filter (if required). Note tat we use a dt value of 1 (day) and not the value of
        # Model.__DELTA_T.

        if self.__use_data:
            points = MerweScaledSigmaPoints(Model.__N_FILTERED,
                                            1e-3,  # Alpha value (usually a small positive value like 1e-3).
                                            2,  # Beta value (a value of 2 is optimal for a Gaussian distribution).
                                            0,  # Kappa value (usually, either 0 or 3-n).
                                            )

            self.__ukf = UnscentedKalmanFilter(Model.__N_FILTERED, Model.__N_MEASURED, 1, self.__h, Model.__f, points)

            self.__ukf.x = np.array([self.__data_i(0), self.__data_r(0), self.__data_d(0),
                                     self.__beta, self.__gamma, self.__mu])

        # Reset our data (if requested).

        if self.__use_data:
            self.__data_s_values = np.array([self.__data_s(0)])
            self.__data_i_values = np.array([self.__data_i(0)])
            self.__data_r_values = np.array([self.__data_r(0)])
            self.__data_d_values = np.array([self.__data_d(0)])

        # Reset our predicted/estimated values.

        self.__s_values = np.array([self.__s_value()])
        self.__i_values = np.array([self.__i_value()])
        self.__r_values = np.array([self.__r_value()])
        self.__d_values = np.array([self.__d_value()])

        # Reset our estimated SIRD model parameters.

        self.__beta_values = np.array([self.__beta])
        self.__gamma_values = np.array([self.__gamma])
        self.__mu_values = np.array([self.__mu])

    def run(self, batch_filter=True, nb_of_days=100):
        """
        Run our SIRD model for the given number of days, taking advantage of the data (if requested) to estimate the
        values of β, γ and μ.
        """

        # Make sure that we were given a valid number of days.

        if not isinstance(nb_of_days, int) or nb_of_days <= 0:
            sys.exit('Error: \'nb_of_days\' must be an integer value greater than zero.')

        # Run our SIRD simulation.

        if batch_filter:
            print('To be done...')
        else:
            for i in range(1, nb_of_days + 1):
                # Compute our predicted/estimated state by computing our SIRD model / Unscented Kalman filter for one
                # day.

                if self.__use_data and self.__data_available(i):
                    self.__ukf.predict(model_self=self)
                    self.__ukf.update(np.array([self.__data_i(i), self.__data_r(i), self.__data_d(i)]))

                    self.__x = self.__ukf.x[:3]
                    self.__beta = self.__ukf.x[3]
                    self.__gamma = self.__ukf.x[4]
                    self.__mu = self.__ukf.x[5]
                else:
                    for j in range(1, Model.__NB_OF_STEPS + 1):
                        self.__x = Model.__f(self.__x, Model.__DELTA_T, model_self=self, with_ukf=False)

                # Keep track of our data (if requested).

                if self.__use_data:
                    if self.__data_available(i):
                        self.__data_s_values = np.append(self.__data_s_values, self.__data_s(i))
                        self.__data_i_values = np.append(self.__data_i_values, self.__data_i(i))
                        self.__data_r_values = np.append(self.__data_r_values, self.__data_r(i))
                        self.__data_d_values = np.append(self.__data_d_values, self.__data_d(i))
                    else:
                        self.__data_s_values = np.append(self.__data_s_values, math.nan)
                        self.__data_i_values = np.append(self.__data_i_values, math.nan)
                        self.__data_r_values = np.append(self.__data_r_values, math.nan)
                        self.__data_d_values = np.append(self.__data_d_values, math.nan)

                # Keep track of our predicted/estimated values.

                self.__s_values = np.append(self.__s_values, self.__s_value())
                self.__i_values = np.append(self.__i_values, self.__i_value())
                self.__r_values = np.append(self.__r_values, self.__r_value())
                self.__d_values = np.append(self.__d_values, self.__d_value())

                # Keep track of our estimated SIRD model parameters.

                self.__beta_values = np.append(self.__beta_values, self.__beta)
                self.__gamma_values = np.append(self.__gamma_values, self.__gamma)
                self.__mu_values = np.append(self.__mu_values, self.__mu)

    def plot(self, figure=None, two_axes=False):
        """
        Plot the results using five subplots for 1) S, 2) I and R, 3) D, 4) β, and 5) γ and μ. In each subplot, we plot
        the data (if requested) as bars and the computed value as a line.
        """

        days = range(self.__s_values.size)
        nrows = 5 if self.__use_data else 3
        ncols = 1

        if figure is None:
            show_figure = True
            figure, axes = plt.subplots(nrows, ncols, figsize=Model.__FIG_SIZE)
        else:
            figure.clf()

            show_figure = False
            axes = figure.subplots(nrows, ncols)

        figure.canvas.set_window_title('SIRD model fitted to data' if self.__use_data else 'Wikipedia SIRD model')

        # First subplot: S.

        axis1 = axes[0]
        axis1.plot(days, self.__s_values, Model.__S_COLOR, label='S')
        axis1.legend(loc='best')
        if self.__use_data:
            axis2 = axis1.twinx() if two_axes else axis1
            axis2.bar(days, self.__data_s_values, color=Model.__S_COLOR, alpha=Model.__DATA_ALPHA)
            data_s_range = self.__population - min(self.__data_s_values)
            data_block = 10 ** (math.floor(math.log10(data_s_range)) - 1)
            s_values_shift = data_block * math.ceil(data_s_range / data_block)
            axis2.set_ylim(min(min(self.__s_values), self.__population - s_values_shift), self.__population)

        # Second subplot: I and R.

        axis1 = axes[1]
        axis1.plot(days, self.__i_values, Model.__I_COLOR, label='I')
        axis1.plot(days, self.__r_values, Model.__R_COLOR, label='R')
        axis1.legend(loc='best')
        if self.__use_data:
            axis2 = axis1.twinx() if two_axes else axis1
            axis2.bar(days, self.__data_i_values, color=Model.__I_COLOR, alpha=Model.__DATA_ALPHA)
            axis2.bar(days, self.__data_r_values, color=Model.__R_COLOR, alpha=Model.__DATA_ALPHA)

        # Third subplot: D.

        axis1 = axes[2]
        axis1.plot(days, self.__d_values, Model.__D_COLOR, label='D')
        axis1.legend(loc='best')
        if self.__use_data:
            axis2 = axis1.twinx() if two_axes else axis1
            axis2.bar(days, self.__data_d_values, color=Model.__D_COLOR, alpha=Model.__DATA_ALPHA)

        # Fourth subplot: β.

        if self.__use_data:
            axis1 = axes[3]
            axis1.plot(days, self.__beta_values, Model.__BETA_COLOR, label='β')
            axis1.legend(loc='best')

        # Fourth subplot: γ and μ.

        if self.__use_data:
            axis1 = axes[4]
            axis1.plot(days, self.__gamma_values, Model.__GAMMA_COLOR, label='γ')
            axis1.plot(days, self.__mu_values, Model.__MU_COLOR, label='μ')
            axis1.legend(loc='best')

        plt.xlabel('time (day)')

        if show_figure:
            plt.show()

    def movie(self, filename):
        """
        Generate, if using the data, a movie showing the evolution of our SIRD model throughout time.
        """

        if self.__use_data:
            data_size = Model.__DATA.shape[0]
            figure = plt.figure(figsize=Model.__FIG_SIZE)
            backend = matplotlib.get_backend()
            writer = manimation.writers['ffmpeg']()

            matplotlib.use("Agg")

            with writer.saving(figure, filename, 96):
                for i in range(1, data_size + 1):
                    print('Processing frame #', i, '/', data_size, '...', sep='')

                    self.__data = Model.__DATA[:i]

                    self.reset()
                    self.run()
                    self.plot(figure=figure)

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
    # Create an instance of the SIRD model, asking for the data to be used.

    m = Model()

    # Run our SIRD model and plot its S, I, R and D values, together with the data.

    m.run()
    m.plot()
