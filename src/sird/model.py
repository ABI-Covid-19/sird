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
    Note that N = S+I+R+D and that we have data from the MoH for I, R and D. So, there is no need to compute S as such.
    """

    __NZ_POPULATION = 5000000
    __N_FILTERED = 6  # Number of state variables to filter (I, R, D, β, γ and μ).
    __N_MEASURED = 3  # Number of measured variables (I, R and D).
    __I_ERROR = 2  # The MoH has, on occasion, reported up to 2 people having been wrongly categorised as infected.
    __R_ERROR = 0.001  # People either recover or they don't, so no errors possible (hence a small value).
    __D_ERROR = 0.001  # People either die or they don't, so no errors possible (hence a small value).
    __BETA_ERROR = 0.001  # We don't determine β ourselves, so no errors possible (hence a small value).
    __GAMMA_ERROR = 0.001  # We don't determine γ ourselves, so no errors possible (hence a small value).
    __MU_ERROR = 0.001  # We don't determine μ ourselves, so no errors possible (hence a small value).
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
    __MOH_DATA = None
    __TEST_DATA = np.array([[1, 0, 0],
                            [1.5667447357441555, 0.06921980348886003, 0.01847169179266075],
                            [2.3662720197300278, 0.1685536217249742, 0.04527656225030958],
                            [3.6206872946691826, 0.33371846307081016, 0.09071801659873499],
                            [5.611643571852583, 0.5890803227161199, 0.15988438320683113],
                            [8.95695179474483, 0.9235533953255841, 0.2682690751188354],
                            [13.224248061552533, 1.5769854300348445, 0.41640253159042384],
                            [21.961640330074868, 2.4591044125683483, 0.6056959798510568],
                            [32.06808740948614, 3.757381606738576, 1.0469085995708567],
                            [48.20473243272773, 6.117181247660241, 1.6216739819962123],
                            [74.0718055138581, 9.314259376606094, 2.5202496677981627],
                            [108.58251886325525, 14.624623865182986, 3.9910712801874895],
                            [169.42583136533926, 22.074853032631736, 5.901098733113967],
                            [265.742852860697, 34.081228545225805, 9.510047374773798],
                            [407.93341568444754, 54.14030905433646, 13.644726980498538],
                            [609.7347666475375, 79.61420745436087, 21.388046772002802],
                            [981.5526764839108, 118.54388030390376, 33.87138964590819],
                            [1483.2136066963064, 195.11440821505505, 49.54110418789029],
                            [2165.746595796803, 300.3302081377656, 74.93265751716606],
                            [3484.5081128243814, 435.0467821977761, 115.92907981918111],
                            [5238.855791600395, 693.9350836987533, 184.48775026466242],
                            [8594.736965627957, 1010.9483840152042, 294.1319099169368],
                            [12616.816530457778, 1657.6319048305963, 420.81221245964224],
                            [19850.50879907105, 2488.6514409437855, 686.7540451380587],
                            [30464.024857623324, 3740.7269804319917, 1035.6341371371557],
                            [48756.69371961697, 5686.25321103713, 1593.107379632083],
                            [67160.95343488482, 8860.234012671639, 2380.4894776092897],
                            [103937.25817216701, 13651.589733717426, 3574.6348159088575],
                            [148805.23657739218, 20900.822921330877, 5888.015593085641],
                            [243208.77619174452, 32014.60059441361, 8808.08518402749],
                            [345859.46694566583, 49058.73720894599, 12678.11112461509],
                            [516858.06147276034, 73852.66910846051, 19386.65919732255],
                            [752382.116466831, 104762.08048831903, 28418.319456911988],
                            [1006434.7959467549, 154107.248891914, 40485.67033560041],
                            [1379861.5219368944, 226527.3154163644, 59194.07745224217],
                            [1660520.6833263559, 309187.99797869206, 84484.01765840276],
                            [2168953.9199777283, 416391.38941371645, 108215.42800478054],
                            [2513479.177592904, 549960.6677799248, 145366.79834811768],
                            [2644842.7617042656, 683312.6094845879, 169868.4480289838],
                            [2747419.4467755137, 825836.8923803412, 224279.90676944365],
                            [2983346.7760959617, 944063.278435433, 264238.0604555801],
                            [3005470.3975589946, 1088951.6464365113, 301300.93182449834],
                            [2834538.113495188, 1300200.8970516494, 345413.4878034352],
                            [2695912.449128643, 1404728.7012084702, 402875.8634173979],
                            [2748040.7766060038, 1656696.766557987, 422865.0108890923],
                            [2729090.428803739, 1857596.79589064, 466276.52399655146],
                            [2353240.356370767, 1885363.3714859134, 514178.74030511343],
                            [2400101.0753614996, 2004731.2163979202, 551928.8624248004],
                            [2104663.7815802745, 2167065.8949705083, 624606.2216115252],
                            [2129577.341436245, 2106574.005604411, 652796.4298393297],
                            [1940138.2328409287, 2431614.8876683726, 617950.4890557947],
                            [1838891.9249776204, 2407260.784342047, 683402.8754182835],
                            [1666537.6174757534, 2530721.2188704284, 699638.1965465001],
                            [1539486.546820981, 2660196.1335526854, 673507.6938665851],
                            [1404354.4642849765, 2832117.362138304, 791249.9891557647],
                            [1317977.6661836847, 2949530.235288108, 789390.6251701905],
                            [1281423.7539783423, 3009395.79030369, 778155.0159239178],
                            [1174181.1696829263, 2949131.4559824276, 794272.5508365985],
                            [1142644.3484378685, 3040202.1978683383, 841598.8359156536],
                            [1051023.3500837423, 3167049.325229189, 830873.0467244339],
                            [1012518.68082894, 2958869.4363810616, 850089.9504926506],
                            [920610.0937517509, 3148803.6702540303, 822328.171820635],
                            [842688.8071564534, 3426165.63109485, 854285.9828393349],
                            [783965.3645178635, 3303377.9958183565, 870540.1031161945],
                            [747797.0140527481, 3293855.5340477778, 925783.9302227726],
                            [661593.3522905209, 3483981.4113144074, 922332.7268192499],
                            [649852.3969680279, 3263617.0054212147, 921994.6153312869],
                            [625350.9128855707, 3590348.117570474, 890272.3796769126],
                            [540690.2769300882, 3660246.5648675263, 966769.5704270509],
                            [538866.9915415694, 3538463.0737157874, 924797.3028936745],
                            [509396.76699295524, 3817478.645948479, 943308.4109891551],
                            [466388.4224254671, 3487576.25889595, 960583.0859665655],
                            [433253.45317020913, 3419414.7490688874, 1027586.511227441],
                            [393561.2362138599, 3505394.0723285964, 961820.179672774],
                            [346984.75610025425, 3557211.958979982, 996827.5980372417],
                            [338757.4788756371, 3464403.612687498, 994837.3201408284],
                            [323011.10317336937, 3664024.0229555303, 1038290.183057335],
                            [308708.73434662295, 3706203.2077737427, 1004058.6965580018],
                            [289210.5268562516, 3719953.0441250843, 1055220.236994065],
                            [263797.3793375069, 3649060.569564463, 1041966.4104980443],
                            [251431.53227567885, 3781816.592512551, 1009758.1519958606],
                            [233855.40798258505, 3703540.507386333, 1042943.6729219066],
                            [219026.61257178595, 3917106.5501983464, 991096.3705790923],
                            [196724.49674745032, 3868268.4951185263, 1006245.8742952407],
                            [192519.33061133113, 3770619.917976897, 1016312.0769050531],
                            [181616.84572149793, 3693837.5548758283, 1035885.262225994],
                            [157543.7465577007, 3877902.1722993655, 1074196.900689952],
                            [155086.74408417087, 3912408.501739155, 1035727.5856412079],
                            [137261.98424313142, 3901548.6358389626, 1086155.3578841595],
                            [129401.46314027767, 3915901.9719678783, 1057462.0247907385],
                            [124219.53653570119, 3814333.829105103, 1057290.6235961074],
                            [118337.09416691639, 3817347.9428325826, 971963.9142922793],
                            [106842.47605512936, 3822792.4599537994, 1020168.7705350053],
                            [96984.60291766284, 3850210.357992408, 1024396.8822907768],
                            [93287.69844311499, 3827334.3515421385, 1075093.6219964027],
                            [88608.16274506014, 3722057.5486506484, 1072781.4736447907],
                            [80737.88659488935, 4018373.355425966, 1047325.6145646736],
                            [80753.41545869978, 3673339.6782685057, 1027149.0284963408],
                            [71468.50411483257, 3999072.2108796923, 1001537.6969717704],
                            [65462.806931105675, 3828660.5137182437, 1069427.0427910818],
                            [60686.37540621936, 3869905.7312526084, 1047891.1000049697]])

    class Use(Enum):
        WIKIPEDIA = auto()
        MOH_DATA = auto()
        TEST_DATA = auto()

    def __init__(self, use=Use.MOH_DATA, max_data=-1):
        """
        Initialise our Model object.
        """

        # Retrieve the MoH data (if requested).

        if use == Model.Use.MOH_DATA and Model.__MOH_DATA is None:
            Model.__MOH_DATA = pd.read_csv('https://bit.ly/3d5eCIq', usecols=[7, 9, 11])

        if use == Model.Use.MOH_DATA:
            self.__data = Model.__MOH_DATA
        else:
            if use == Model.Use.TEST_DATA:
                self.__data = Model.__TEST_DATA
            else:
                self.__data = None

        if self.__data is not None and max_data != -1:
            self.__data = self.__data[:max_data]

        # Keep track of whether to use the MoH/test data.

        self.__use_moh_data = use == Model.Use.MOH_DATA
        self.__use_test_data = use == Model.Use.TEST_DATA

        self.__use_data = self.__use_moh_data or self.__use_test_data

        # Declare some internal variables (that will then be initialised through our call to reset()).

        self.__beta = None
        self.__gamma = None
        self.__mu = None

        self.__ukf = None

        self.__x_p = None
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

    def __data_s(self, day):
        """
        Return the MoH/test S value for the given day.
        """

        return Model.__NZ_POPULATION - self.__data_i(day) - self.__data_r(day) - self.__data_d(day)

    def __data_x(self, day, index):
        """
        Return the MoH/test I/R/D value for the given day.
        """

        if type(day) is int:
            return self.__data.iloc[day][index] if self.__use_moh_data \
                else self.__data[day][index] if self.__use_test_data \
                else math.nan
        else:
            floor_day = math.floor(day)
            floor_day_data_x = self.__data_x(floor_day, index)

            return floor_day_data_x + (day - floor_day) * (self.__data_x(math.ceil(day), index) - floor_day_data_x)

    def __data_i(self, day):
        """
        Return the MoH/test I value for the given day.
        """

        return self.__data_x(day, 2 if self.__use_moh_data else 0)

    def __data_r(self, day):
        """
        Return the MoH/test R value for the given day.
        """

        return self.__data_x(day, 0 if self.__use_moh_data else 1)

    def __data_d(self, day):
        """
        Return the MoH/test D value for the given day.
        """

        return self.__data_x(day, 1 if self.__use_moh_data else 2)

    def __data_available(self, day):
        """
        Return whether some data is available for the given day.
        """

        return day <= self.__data.shape[0] - 1 if self.__use_data else False

    def __s_value(self):
        """
        Return the S value based on the values of I, R, D and N.
        """

        return self.__n - self.__x_p.sum()

    def __i_value(self):
        """
        Return the I value.
        """

        return self.__x_p[0]

    def __r_value(self):
        """
        Return the R value.
        """

        return self.__x_p[1]

    def __d_value(self):
        """
        Return the D value.
        """

        return self.__x_p[2]

    def reset(self):
        """
        Reset our SIRD model.
        """

        # Reset β, γ and μ to the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        self.__beta = 0.4
        self.__gamma = 0.035
        self.__mu = 0.005

        # Reset I, R and D to the MoH data at day 0 or the values mentioned on Wikipedia (see https://bit.ly/2VMvb6h).

        if self.__use_data:
            points = MerweScaledSigmaPoints(Model.__N_FILTERED,
                                            1e-3,  # Alpha value (usually a small positive value like 1e-3).
                                            2,  # Beta value (a value of 2 is optimal for a Gaussian distribution).
                                            0,  # Kappa value (usually, either 0 or 3-n).
                                            )

            self.__ukf = UnscentedKalmanFilter(Model.__N_FILTERED, Model.__N_MEASURED, Model.__DELTA_T, self.__h,
                                               Model.__f, points)

            self.__ukf.x = np.array(
                [self.__data_i(0), self.__data_r(0), self.__data_d(0), self.__beta, self.__gamma, self.__mu])
            self.__ukf.P = np.diag([Model.__I_ERROR ** 2, Model.__R_ERROR ** 2, Model.__D_ERROR ** 2,
                                    Model.__BETA_ERROR ** 2, Model.__GAMMA_ERROR ** 2, Model.__MU_ERROR ** 2])

            self.__x_p = np.array([self.__data_i(0), self.__data_r(0), self.__data_d(0)])
            self.__n = Model.__NZ_POPULATION
        else:
            self.__x_p = np.array([3, 0, 0])
            self.__n = 1000

        # Reset our MoH data and simulation values.

        if self.__use_data:
            self.__data_s_values = np.array([self.__data_s(0)])
            self.__data_i_values = np.array([self.__data_i(0)])
            self.__data_r_values = np.array([self.__data_r(0)])
            self.__data_d_values = np.array([self.__data_d(0)])

        self.__s_values = np.array([self.__s_value()])
        self.__i_values = np.array([self.__i_value()])
        self.__r_values = np.array([self.__r_value()])
        self.__d_values = np.array([self.__d_value()])

        self.__beta_values = np.array([self.__beta])
        self.__gamma_values = np.array([self.__gamma])
        self.__mu_values = np.array([self.__mu])

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

        k = None

        for i in range(nb_of_days):
            # Compute our predicted state, i.e. compute the SIRD model for one day.

            for j in range(1, Model.__NB_OF_STEPS + 1):
                k = i + j * Model.__DELTA_T

                if self.__use_data and self.__data_available(k):
                    self.__ukf.predict(model_self=self)
                    self.__ukf.update(np.array([self.__data_i(k), self.__data_r(k), self.__data_d(k)]))

                    self.__x_p = self.__ukf.x[:3]
                    self.__beta = self.__ukf.x[3]
                    self.__gamma = self.__ukf.x[4]
                    self.__mu = self.__ukf.x[5]
                else:
                    self.__x_p = Model.__f(self.__x_p, Model.__DELTA_T, model_self=self, with_ukf=False)

            # Update our MoH data (if requested) and simulation values.

            if self.__use_data:
                if self.__data_available(k):
                    self.__data_s_values = np.append(self.__data_s_values, self.__data_s(k))
                    self.__data_i_values = np.append(self.__data_i_values, self.__data_i(k))
                    self.__data_r_values = np.append(self.__data_r_values, self.__data_r(k))
                    self.__data_d_values = np.append(self.__data_d_values, self.__data_d(k))
                else:
                    self.__data_s_values = np.append(self.__data_s_values, math.nan)
                    self.__data_i_values = np.append(self.__data_i_values, math.nan)
                    self.__data_r_values = np.append(self.__data_r_values, math.nan)
                    self.__data_d_values = np.append(self.__data_d_values, math.nan)

            self.__s_values = np.append(self.__s_values, self.__s_value())
            self.__i_values = np.append(self.__i_values, self.__i_value())
            self.__r_values = np.append(self.__r_values, self.__r_value())
            self.__d_values = np.append(self.__d_values, self.__d_value())

            self.__beta_values = np.append(self.__beta_values, self.__beta)
            self.__gamma_values = np.append(self.__gamma_values, self.__gamma)
            self.__mu_values = np.append(self.__mu_values, self.__mu)

    def plot(self, fig=None, two_axes=False):
        """
        Plot the results using five subplots for 1) S, 2) I and R, 3) D, 4) β, and 5) γ and μ. In each subplot, we plot
        the MoH data (if requested) as bars and the computed value as a line.
        """

        days = range(self.__s_values.size)
        nrows = 5 if self.__use_data else 3
        ncols = 1

        if fig is None:
            show_fig = True
            fig, ax = plt.subplots(nrows, ncols, figsize=Model.__FIG_SIZE)
        else:
            fig.clf()

            show_fig = False
            ax = fig.subplots(nrows, ncols)

        if self.__use_moh_data:
            fig.canvas.set_window_title('SIRD model fitted to MoH data')
        else:
            if self.__use_test_data:
                fig.canvas.set_window_title('SIRD model fitted to test data')
            else:
                fig.canvas.set_window_title('Wikipedia SIRD model')

        # First subplot: S.

        ax1 = ax[0]
        ax1.plot(days, self.__s_values, Model.__S_COLOR, label='S')
        ax1.legend(loc='best')
        if self.__use_data:
            ax2 = ax1.twinx() if two_axes else ax1
            ax2.bar(days, self.__data_s_values, color=Model.__S_COLOR, alpha=Model.__DATA_ALPHA,
                    label='MoH S' if self.__use_moh_data else 'Test S')
            data_s_range = Model.__NZ_POPULATION - min(self.__data_s_values)
            data_block = 10 ** (math.floor(math.log10(data_s_range)) - 1)
            s_values_shift = data_block * math.ceil(data_s_range / data_block)
            ax2.set_ylim(Model.__NZ_POPULATION - s_values_shift, Model.__NZ_POPULATION)

        # Second subplot: I and R.

        ax1 = ax[1]
        ax1.plot(days, self.__i_values, Model.__I_COLOR, label='I')
        ax1.plot(days, self.__r_values, Model.__R_COLOR, label='R')
        ax1.legend(loc='best')
        if self.__use_data:
            ax2 = ax1.twinx() if two_axes else ax1
            ax2.bar(days, self.__data_i_values, color=Model.__I_COLOR, alpha=Model.__DATA_ALPHA,
                    label='MoH I' if self.__use_moh_data else 'Test I')
            ax2.bar(days, self.__data_r_values, color=Model.__R_COLOR, alpha=Model.__DATA_ALPHA,
                    label='MoH R' if self.__use_moh_data else 'Test R')

        # Third subplot: D.

        ax1 = ax[2]
        ax1.plot(days, self.__d_values, Model.__D_COLOR, label='D')
        ax1.legend(loc='best')
        if self.__use_data:
            ax2 = ax1.twinx() if two_axes else ax1
            ax2.bar(days, self.__data_d_values, color=Model.__D_COLOR, alpha=Model.__DATA_ALPHA,
                    label='MoH D' if self.__use_moh_data else 'Test D')

        # Fourth subplot: β.

        if self.__use_data:
            ax1 = ax[3]
            ax1.plot(days, self.__beta_values, Model.__BETA_COLOR, label='β')
            ax1.legend(loc='best')

        # Fourth subplot: γ and μ.

        if self.__use_data:
            ax1 = ax[4]
            ax1.plot(days, self.__gamma_values, Model.__GAMMA_COLOR, label='γ')
            ax1.plot(days, self.__mu_values, Model.__MU_COLOR, label='μ')
            ax1.legend(loc='best')

        plt.xlabel('time (day)')

        if show_fig:
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

    def movie(self, filename):
        if self.__use_data:
            if self.__use_moh_data:
                data_size = Model.__MOH_DATA.shape[0]
            else:
                data_size = Model.__TEST_DATA.shape[0]

            fig = plt.figure(figsize=Model.__FIG_SIZE)
            backend = matplotlib.get_backend()
            writer = manimation.writers['ffmpeg'](15)

            matplotlib.use("Agg")

            with writer.saving(fig, filename, 96):
                for i in range(1, data_size + 1):
                    print('Processing frame #', i, '/', data_size, '...', sep='')

                    if self.__use_moh_data:
                        self.__data = Model.__MOH_DATA
                    else:
                        self.__data = Model.__TEST_DATA

                    self.__data = self.__data[:i]

                    self.reset()
                    self.run()
                    self.plot(fig=fig)

                    writer.grab_frame()

                print('All done!')

            matplotlib.use(backend)


if __name__ == '__main__':
    # Create an instance of the SIRD model, asking for the MoH data to be used.

    m = Model()

    # Run the model and plot its S, I, R and D values, together with the MoH data.

    m.run()
    m.plot()
