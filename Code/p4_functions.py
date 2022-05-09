# Imports
import cmath
import logging

import numpy as np
import pandas as pd
from scipy import constants


# -------------- Function Definitions --------------#
def test_function(x: int):
    """Just to make sure my syntax is ok 😊"""
    print(f"It Works {x} times!")


def create_training_data(num_samples):
    """Configure a Pandas dataframe with training data

    Input:
        num_samples: integer corresponding to how many rows will be in the returned data frame

    Return:
        data: Pandas DataFrame, each row is a sample. Columns are:
            'mag_rad': 101x101 matrix, each item is a complex number corresponding to field data
            'width': width of waveguide in meters (float)
            'height': height of waveguide in meters (float)
            'm': Mode number 'm' (int)
            'n': Mode number 'n' (int)
            'mode': TE or TM mode (0 or 1 respectively)
            'freq': Frequency of operation (float)
            'component': Ex, Ey, Ez, Hx, Hy, Hz (0 to 5 respectively)
    """

    # How many samples to generate?
    num_samples = int(num_samples)

    # Statistical ranges to use
    width_min = 0.1 * constants.centi
    width_max = 10.0 * constants.centi
    height_min = 0.05 * constants.centi
    height_max = 5.0 * constants.centi
    m_min = 1
    m_max = 3
    n_min = 1
    n_max = 3
    mode_min = 0
    mode_max = 1
    # freq_min = 1.5 * constants.mega # Not needed, compute via cutoff frequency
    freq_max = 150.0 * constants.giga
    component_min = 0
    component_max = 5

    column_names = ['mag_rad', 'width', 'height', 'm', 'n', 'mode', 'freq', 'component']

    data = pd.DataFrame(columns=column_names)  # Empty data frame to populate

    for i in range(num_samples):
        # Set random values
        # Uniform sampling across ranges
        width = float(np.random.uniform(width_min, width_max))
        height = float(np.random.uniform(height_min, height_max))
        m = int(np.random.uniform(m_min, m_max + 1))  # Max +1 b/c rand.uniform is max exclusive
        n = int(np.random.uniform(n_min, n_max + 1))
        mode = int(np.random.uniform(mode_min, mode_max + 1))
        # From pozar Ch. 3, there is no TE00, TM00, TM01, TM10 mode
        if m == 0 and n == 0:  # DO NOT ALLOW THIS, TE00/TM00
            m = int(np.random.uniform(1, m_max + 1))
            n = int(np.random.uniform(1, n_max + 1))
        if (mode == 1) and (m == 0 or n == 0):  # TM01 or TM10
            m = int(np.random.uniform(1, m_max + 1))
            n = int(np.random.uniform(1, n_max + 1))

        # Need to determine cutoff frequency
        f_cutoff = (constants.c / (2 * np.pi)) * np.sqrt((m * np.pi / width) ** 2 + (n * np.pi / height) ** 2)

        # Frequency minimum is the cutoff frequency
        freq = float(np.random.uniform(f_cutoff, freq_max))

        component = int(np.random.uniform(component_min, component_max + 1))

        # BIG CHANGE HERE
        # Avoid zeros. If mode=0 and component=2 (TE, Ez) then swap to TM
        # Same for TM, Hz (swap to TE)
        if (mode == 0) and (component == 2):
            mode = 1
        if (mode == 1) and (component == 5):
            mode = 0
        # ----------------------

        # Calculate basic constants
        # Assumption here: filled with a vacuum
        k = 2 * np.pi * freq * np.sqrt(constants.mu_0 * constants.epsilon_0)
        kc = np.sqrt((((m * np.pi) / width) ** 2) + (((n * np.pi) / height) ** 2))
        beta = np.sqrt((k ** 2) + (kc ** 2))
        # Assumption: Amplitude constants A are always 1

        value = np.zeros((101, 101), dtype=complex)

        # Pre-init variables due to try/except
        x = None
        x_coord = None
        y = None
        y_coord = None

        # Have to set X and Y!
        try:
            for x_coord in range(101):
                for y_coord in range(101):
                    x = float((x_coord / 100) * width)
                    y = float((y_coord / 100) * height)
                    # Determine equation based on component and mode
                    # Assume waveguide is vacuum filled
                    if mode == 0:  # TE mode
                        if component == 0:  # Ex
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(((
                                        (1j * 2 * np.pi * freq * constants.mu_0 * n * np.pi)
                                        / (height * (kc ** 2)))) * cmath.cos((m * np.pi * x) / width)
                                                                  * cmath.sin((n * np.pi * y) / height))
                        elif component == 1:  # Ey
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(((
                                        (-1j * 2 * np.pi * freq * constants.mu_0 * m * np.pi)
                                        / (width * (kc ** 2)))) * cmath.sin((m * np.pi * x) / width)
                                                                  * cmath.cos((n * np.pi * y) / height))
                        elif component == 2:  # Ez
                            value[x_coord, y_coord] = 0
                        elif component == 3:  # Hx
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(
                                    (1j * beta * m * np.pi / width * (kc ** 2)) * cmath.sin(
                                        (m * np.pi * x) / width) * cmath.cos((n * np.pi * y) / height))
                        elif component == 4:  # Hy
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(
                                    (1j * beta * n * np.pi / height * (kc ** 2)) * cmath.cos(
                                        (m * np.pi * x) / width) * cmath.sin((n * np.pi * y) / height))
                        elif component == 5:  # Hz
                            value[x_coord, y_coord] = complex(
                                cmath.cos((m * np.pi * x) / width) * cmath.cos((n * np.pi * y) / height))
                        else:
                            print("ERROR: COMPONENT NOT SET CORRECTLY")
                    elif mode == 1:  # TM mode
                        if component == 0:  # Ex
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(
                                    (-1j * beta * m * np.pi / width * (kc ** 2)) * cmath.cos(
                                        (m * np.pi * x) / width) * cmath.sin((n * np.pi * y) / height))
                        elif component == 1:  # Ey
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(
                                    (-1j * beta * n * np.pi / height * (kc ** 2)) * cmath.sin(
                                        (m * np.pi * x) / width) * cmath.cos((n * np.pi * y) / height))
                        elif component == 2:  # Ez
                            value[x_coord, y_coord] = complex(
                                cmath.sin((m * np.pi * x) / width) * cmath.sin((n * np.pi * y) / height))
                        elif component == 3:  # Hx
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(((
                                        (1j * 2 * np.pi * freq * constants.epsilon_0 * n * np.pi)
                                        / (height * (kc ** 2)))) * cmath.sin((m * np.pi * x) / width)
                                                                  * cmath.cos((n * np.pi * y) / height))
                        elif component == 4:  # Hy
                            if (m == 0) and (n == 0):
                                value[x_coord, y_coord] = 0
                            else:
                                value[x_coord, y_coord] = complex(((
                                        (-1j * 2 * np.pi * freq * constants.epsilon_0 * m * np.pi)
                                        / (width * (kc ** 2)))) * cmath.cos((m * np.pi * x) / width)
                                                                  * cmath.sin((n * np.pi * y) / height))
                        elif component == 5:  # Hz
                            value[x_coord, y_coord] = 0
                        else:
                            print("ERROR: COMPONENT NOT SET CORRECTLY")
                    else:
                        print("ERROR CONDITION - MODE NOT SET CORRECTLY")

        except Exception as e:
            logging.exception(e)
            print("ERROR HERE!")
            print(f"i: {i}")
            print(f"x: {x}")
            print(f"x_coord: {x_coord}")
            print(f"y: {y}")
            print(f"y_coord: {y_coord}")
            print(f"width: {width}")
            print(f"height: {height}")
            print(f"m: {m}")
            print(f"n: {n}")
            print(f"mode: {mode}")
            print(f"freq: {freq}")
            print(f"component: {component}")
            
        # Add noise in here
        # Remove this line if you want to use noiseless
        value = np.random.exponential(1, [101, 101]) * value

        data_temp = [[value, width, height, m-1, n-1, mode, freq, component]]
        df_temp = pd.DataFrame(data_temp, columns=column_names)
        data = pd.concat([data, df_temp])

    data.index = pd.RangeIndex(len(data.index))
    return data


def add_noise(data, std, noise_type):
    """Add noise to a matrix

    This function takes in the mag/rad complex matrix of 101x101 size and multiplies by the specified noise.

    :param data: Input a 101x101 matrix with real or complex data
    :param std: standard deviation used in analysis. Multiplies by mean
    :param noise_type: Type of noise to be added. Options are: "normal" (none else at the moment)
    :return: 101x101 matrix. If no issues, it will return data with noise multiplied.
    If there are issues, it will still return the original data to prevent higher level problems
    """
    # Take in 101x101 data matrix, apply noise of some type to it

    # NOISE: How much? What direction? (Random? Up and down? Apply to each pixel?)
    # What's the mean? Std dist?

    # create array of noise values and multiply by final "value" array

    if data.shape != (101, 101):
        print("DATA NOT CORRECT SIZE")
        return data

    # Determine mean of the real and imaginary parts
    # mean_real = np.mean(data.real)
    # mean_imag = np.mean(data.imag)
    mean_data = np.mean(abs(data))
    mean_overall = 1

    # Noise uses:
    # Mean = mean of the data set
    # Std is equal to the mean, but multiplied by some value

    if noise_type == "normal":
        noise_matrix = np.random.normal(mean_overall, std*mean_overall, [101, 101])
        noisy_data = noise_matrix * data
        return noisy_data
    elif noise_type == "exponential":
        # We want the "center" of this to be at zero, then we want the "spread" adjustable

        noise_matrix = np.random.exponential(std, [101, 101])
        noisy_data = noise_matrix * data
        return noisy_data
    else:
        print("INCORRECT TYPE GIVEN")
        return data


def noise_gen_data(input_data: pd.DataFrame, std_deviation: float) -> pd.DataFrame:
    """
    Take in a pd dataframe of any length. Assumes it has a 'mag_rad' column.
    Adds noise per the input parameter

    :param input_data:
    :param std_deviation: value to use for standard deviation
    :return: pd dataframe, no change to structure but noise added to the mag_rad column
    """

    output_data = input_data.copy()

    data_length = len(output_data.index)

    for i in range(data_length):
        output_data['mag_rad'][i] = add_noise(output_data['mag_rad'][i], std_deviation, "exponential")

    return output_data
