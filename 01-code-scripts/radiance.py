# Imports
import re
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import rasterio as rio
import earthpy.mask as em


def add_missing_data(df, start_date, end_date):
    """Adds entries for missing dates and populates
    the data for those dates with NaN values.

    Meant for use with radiance values or cloud mask
    values and assumed dates are columns (not indices).

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with missing dates.

    Returns
    -------
    filled_df : pandas dataframe
        Dataframe with full date range and
        NaN values for data added.

    Example
    -------
        >>> filled = add_missing_data(radiance, '2019-09-01', '2020-04-30')
        >>>
        >>>
    """
    # Create copy of dataframe (avoids changing the original)
    df_copy = df.copy()

    # Transpose dataframe to get dates an indices
    transposed_df = df_copy.transpose()

    # Create pandas class of current indices (not full range)
    transposed_df.index = pd.DatetimeIndex(transposed_df.index)

    # Create index for full date range
    full_date_range_index = pd.date_range(start_date, end_date)

    # Add missing dates, with NaN values for data
    transposed_df_filled = transposed_df.reindex(
        full_date_range_index, fill_value=np.NaN)

    # Transpose dataframe back to pixel IDs as indices
    filled_df = transposed_df_filled.transpose()

    # Change all column names to str (from datetime)
    filled_df.columns = filled_df.columns.strftime("%Y-%m-%d")

    # Return filled dataframe
    return filled_df


def create_date_list(start_date, end_date, date_frequency='D'):
    """Returns a list of dates in YYYY-MM-DD format,
    within the specified range and frequency.

    Parameters
    ----------
    start_date : str
        Date in the following format: 'YYYY-MM-DD'.

    end_date : str
        Date in the following format: 'YYYY-MM-DD'.

    date_frequency : str, optional
        Frequency of dates. Defaults to daily ('D').

    Returns
    -------
    date_list : list
        List of dates between in the range specified,
        including the start and end dates.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Create pandas date range (all days) with start and end date
    date_range = pd.date_range(start_date, end_date, freq=date_frequency)

    # Create list of dates (as strings) formatted to YYYY-MM-DD (daily)
    date_list = [date.strftime("%Y-%m-%d") for date in date_range]

    # Return list of dates
    return date_list


def get_data(radiance_df, year, month, day):
    """Extracts single-day radiance (values or
    cloud mask values) into a dataframe, based on
    the specified date.

    Parameters
    ----------
    radiance_df : pandas dataframe
        Dataframe indexed by pixel ID containing the
        radiance data by date (columns).

    year : str
        Four-digit year (YYYY).

    month : str
        Two-digit month (MM), with leading 0s
        (ex: '01', '02', '03', '10').

    day : str
        Two-digit (MM) month, with leading 0s
        (ex: '01', '02', '03', '10').

    Returns
    -------
    radiance : pandas dataframe
        Dataframe containing the single-day radiance data.

    Example
    -------
        >>> radiance_data = get_radiance_data(radiance_df, year='2019', month='09', day='01')
        >>>
        >>>
        >>>
    """
    # Get single-day radiance data (values or cloud mask) dataframe
    #  that matches the exact date in the input dataframe
    radiance = [radiance_df[[col]]
                for col in radiance_df.columns
                if re.compile(f"^{year}-{month}-{day}$").match(col)][0]

    # Return the single-day radiance dataframe
    return radiance


def get_array(radiance_data, output_rows=18, output_columns=40):
    """Extracts radiance and cloud mask data into a
    correctly-shaped array for a study area.

    Default values meant for specific use with the
    Penn State campus data.

    Parameters
    ----------
    radiance_data : pandas dataframe
        Dataframe containing the radiance data
        (radiance values or cloud mask values).

    output_rows : int, optional
        Number of rows in the output array (used
        in the reshaping). Defaults to 18.

    output_columns : int, optional
        Number of columns in the output array (used
        in the reshaping). Defaults to 40.

    Returns
    -------
    radiance_array : numpy array
        Numpy array containg the formatted radiance data
        (radiance values or cloud mask values).

    Example
    ------
        >>> import pandas as pd
        >>> radiance_df = pd.read_csv(radiance_path)
        >>> radiance_arr = get_array(radiance_df)
    """
    # Convert dataframe to numpy array, reshape array, and transpose array
    # Rows and columns must be flipped in .reshape due to how the data
    #  is read into the dataframe
    radiance_array = radiance_data.to_numpy().reshape((output_columns, output_rows)).transpose()

    # Return correctly-shaped array
    return radiance_array


def store_data(radiance_df, cloud_mask_df, mask_value, dates):
    """Masks and stores daily radiance data
    in a dictionary.

    Parameters
    ----------
    radiance_df : pandas dataframe
        Dataframe containing radiance values, with date
        as column name.

    cloud_mask_df : pandas dataframe
        Dataframe containing radiance cloud mask values,
        with date as column name.

    mask_value : int
        Value indicating cloudy pixel that requires masking.

    dates : list
        List of dates (strings), formatted as 'YYYY-MM-DD'.

    Returns
    -------
    radiance_masked : dictionary
        Dictionary containing masked daily radiance arrays,
        indexed by dictionary['YYYY']['MM']['DD'].

    Example
    -------
        >>>
        >>>
        >>>
    """
    # Create dictionary to store cloud free radiance data
    radiance_masked = {}

    # Loop through all dates in provided date list
    for date in dates:

        # Split date into year/month/day components
        year = date.split('-')[0]
        month = date.split('-')[1]
        day = date.split('-')[2]

        # Add year to dictionary if not existing key
        if year not in radiance_masked.keys():
            radiance_masked[year] = {}

        # Add month dictionary if not existing key within year
        if month not in radiance_masked.get(year).keys():
            radiance_masked[year][month] = {}

        # Get radiance data
        radiance = get_data(
            radiance_df, year=year, month=month, day=day)

        # Get cloud mask data
        cloud_mask = get_data(
            cloud_mask_df, year=year, month=month, day=day)

        # Create array from dataframe
        radiance_array = get_array(radiance)
        cloud_mask_array = get_array(cloud_mask)

        # Create tuple for radiance data and cloud mask
        radiance_mask_tuple = (radiance_array, cloud_mask_array)

        # Check if array should be masked (includes mask value)
        if mask_value in radiance_mask_tuple[1]:

            # Mask with cloud mask value
            masked_array = em.mask_pixels(
                radiance_mask_tuple[0],
                radiance_mask_tuple[1],
                vals=[mask_value])

        # If no mask value (or contains NaN values)
        else:

            # Assign original array to cloud free
            masked_array = radiance_mask_tuple[0]

        # Store masked array in dictionary, indexed by date
        radiance_masked[year][month][day] = masked_array

    # Return dictionary of masked data
    return radiance_masked


def extract_data(radiance, dates):
    """Returns a list of arrays from a nested dictionary,
    that is indexed by dictionary[Year][Month][Day].

    Meant for intra and inter-month date ranges (both
    continuous and not continuous).

    Parameters
    ----------
    radiance : dict
        Dictionary containing masked daily radiance arrays,
        indexed by dictionary['YYYY']['MM']['DD'].

    dates : list
        List of dates (strings), formatted as 'YYYY-MM-DD'.

    Returns
    -------
    array_list : list
        List of masked radiance arrays.

    Example
    -------
        >>>
        >>>
        >>>
    """
    # Flatten dataframe into dictionary
    radiance_df = json_normalize(radiance)

    # Replace '.' with '-' in column names
    radiance_df.columns = [column.replace(
        '.', '-') for column in radiance_df.columns]

    # Create list of arrays based on date list
    array_list = []

    # Loop through all dates specified
    for date in dates:

        # Extact year/month/day components from date
        year = date.split('-')[0]
        month = date.split('-')[1]
        day = date.split('-')[2]

        # Loop through all columns in flattened dataframe
        for col in radiance_df.columns:

            # Check if date exists within dataframe columns
            if re.compile(f"^{year}-{month}-{day}$").match(col):

                # Add array for specified date to the list of arrays
                array_list.append(radiance_df[col].loc[0])

    # Return list of arrays
    return array_list


def flatten_data(radiance):
    """Extracts radiance arrays for a whole month
    from a dictionary into a list of arrays.

    Meant for use with a single month of data.

    Parameters
    ----------
    radiance : dict
        Dictionary containing one whole month of
        radiance arrays, with days as keys and
        arrays as values.

    Returns
    -------
    array_list : list
        List of numpy arrays (or masked numpy
        arrays) containing masked radiance values.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Create list of arrays from dictionary
    array_list = [radiance.get(key) for key in radiance.keys()]

    # Return list of arrays
    return array_list


def calculate_mean(radiance_data):
    """Calculates the mean over many arrays
    covering the same area.

    Parameters
    ----------
    radiance_data : list
        List of masked numpy arrays (can contain NaN values).

    Returns
    -------
    radiance_stack_mean : numpy array
        Numpy array containing the mean value for each pixel,
        averaged over the number of arrays in the input list.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Create stack of numpy arrays (3d array)
    radiance_stack = np.stack(radiance_data)

    # Get mean value for each pixel, over all arrays (bands)
    radiance_stack_mean = np.nanmean(radiance_stack, axis=0)

    # Return mean array
    return radiance_stack_mean


def export_array(array, output_path, metadata):
    """Exports a numpy array to a GeoTiff.

    Parameters
    ----------
    array : numpy array
        Numpy array to be exported to GeoTiff.

    output_path : str
        Path to the output file (includeing filename).

    metadata : dict
        Dictionary containing the metadata required
        for export.

    Returns
    -------
    output_message : str
        Message indicating success or failure of export.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    try:
        # Write numpy array to GeoTiff
        with rio.open(output_path, 'w', **metadata) as dst:
            dst.write(array, 1)

    except Exception as error:
        output_message = f"ERROR: {error}"

    else:
        output_message = f"Exported {output_path}"

    return print(output_message)
