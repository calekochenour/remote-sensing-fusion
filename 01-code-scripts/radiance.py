# Imports
import re
from collections import ChainMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import rasterio as rio
from rasterio.transform import from_origin
import earthpy.plot as ep
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
        >>> # Define path to radiance data
        >>> psu_radiance_path = os.path.join(
        ...     radiance_directory,
        ...     "PSU_Radiance_RavelOrderF.csv")
        >>> # Read radiance values to dataframe
        >>> psu_radiance_df = pd.read_csv(psu_radiance_path)
        >>> # Add missing data
        >>> psu_radiance_df_filled = add_missing_data(
        ...     psu_radiance_df, '2019-09-01', '2020-04-30')
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
        >>> # Get list of dates (daily) from Sept 2019 to April 2020
        >>> date_list = create_date_list(
        ...     start_date='2019-09-01',
        ...     end_date='2020-04-30')
        >>> # Display number of days
        >>> len(date_list)
        243
        >>> # Display first day in list
        >>> date_list[0]
        '2019-09-01'
        >>> # Display last day in list
        >>> date_list[-1]
        '2020-04-30'
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
        >>> # Define path to radiance data
        >>> psu_radiance_path = os.path.join(
        ...     radiance_directory,
        ...     "PSU_Radiance_RavelOrderF.csv")
        >>> # Read radiance values to dataframe
        >>> psu_radiance_df = pd.read_csv(psu_radiance_path)
        >>> # Get radiance data for September 1, 2019
        >>> radiance_2019_09_01 = get_data(
        ...     psu_radiance_df, year='2019', month='09', day='01')
    """
    # Get single-day radiance data (values or cloud mask) dataframe
    #  that matches the exact date in the input dataframe
    radiance = [radiance_df[[col]]
                for col in radiance_df.columns
                if re.compile(f"^{year}-{month}-{day}$").match(col)][0]

    # Return the single-day radiance dataframe
    return radiance


def get_array(radiance_data, output_rows, output_columns):
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
        >>> # Define path to radiance data
        >>> psu_radiance_path = os.path.join(
        ...     radiance_directory,
        ...     "PSU_Radiance_RavelOrderF.csv")
        >>> # Read radiance values to dataframe
        >>> psu_radiance_df = pd.read_csv(psu_radiance_path)
        >>> # Create array from dataframe
        >>> psu_radiance_arr = get_array(psu_radiance_df, 18, 40)
        >>> # Display type
        numpy.ma.core.MaskedArray
        >>> Display array shape
        >>> psu_radiance_arr.shape
        (18, 40)
    """
    # Convert dataframe to numpy array, reshape array, and transpose array
    # Rows and columns must be flipped in .reshape due to how the data
    #  is read into the dataframe
    radiance_array = radiance_data.to_numpy().reshape((output_columns, output_rows)).transpose()

    # Return correctly-shaped array
    return radiance_array


def store_data(radiance_df, cloud_mask_df, mask_value, array_shape, dates):
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

    array_shape : tuple (of ints)
        Tuple containing the shape (rows, columns) of the
        output numpy arrays.

    dates : list
        List of dates (strings), formatted as 'YYYY-MM-DD'.

    Returns
    -------
    radiance_masked : dictionary
        Dictionary containing masked daily radiance arrays,
        indexed by dictionary['YYYY']['MM']['DD'].

    Example
    -------
        >>> # Define path to radiance data
        >>> psu_radiance_path = os.path.join(
        ...     radiance_directory,
        ...     "PSU_Radiance_RavelOrderF.csv")
        >>> psu_cloud_mask_path = os.path.join(
        ...     radiance_directory,
        ...     "PSU_CloudMask_RavelOrderF.csv")
        >>> # Read radiance data to dataframes
        >>> psu_radiance_df = pd.read_csv(psu_radiance_path)
        >>> psu_cloud_mask_df = pd.read_csv(psu_cloud_mask_path)
        >>> # Add missing data
        >>> psu_radiance_filled = add_missing_data(
        ...     psu_radiance_df, '2019-09-01', '2020-04-30')
        >>> psu_clous_mask_filled = add_missing_data(
        ...     psu_cloud_mask_df, '2019-09-01', '2020-04-30')
        >>> # Create date list
        >>> date_list = create_date_list(
        ...     start_date='2019-09-01',
        ...     end_date='2020-04-30')
        >>> # Store all daily filled values in nested dictionary
        >>> radiance_sept_2019_apr_2020 = store_data(
        ...     psu_radiance_df_filled,
        ...     psu_cloud_mask_df_filled,
        ...     mask_value=100,
        ...     array_shape=(18, 40),
        ...     dates=date_list)
        >>> # Display top-level keys
        >>> radiance_sept_2019_apr_2020.keys()
        dict_keys(['2019', '2020'])
        >>> # Display 2019 keys
        >>> radiance_sept_2019_apr_2020.get('2019').keys()
        dict_keys(['09', '10', '11', '12'])
        >>> Display max radiance on September 15, 2019
        >>> radiance_sept_2019_apr_2020['2019']['09']['15'].max()
        1121.0
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
        radiance_array = get_array(radiance, array_shape[0], array_shape[1])
        cloud_mask_array = get_array(cloud_mask, array_shape[0], array_shape[1])

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
        >>> # Create date list
        >>> date_list = create_date_list(
        ...     start_date='2019-09-01',
        ...     end_date='2020-04-30')
        >>> # Store all daily filled values in nested dictionary
        >>> radiance_sept_2019_apr_2020 = store_data(
        ...     psu_radiance_df_filled,
        ...     psu_cloud_mask_df_filled,
        ...     mask_value=100,
        ...     array_shape=(18, 40),
        ...     dates=date_list)
        # Create date range to extract
        >>> date_range = create_date_list('2019-12-22', '2020-01-10'),
        >>> # Get radiance array for each date into list
        >>> radiance_arrays = extract_data(
        ...     radiance=radiance_sept_2019_apr_2020, dates=date_range)
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
        >>> # Get September 2019 data (all days)
        >>> radiance = radiance_sept_2019_apr_2020.get('2019').get('09')
        >>> # Flatten dictionary to list of arrays
        >>> radiance_arrays = flatten_data(radiance)
        >>> # Display type
        >>> type(radiance_arrays)
        list
        >>> Display number of days
        >>> len(radiance_arrays)
        30
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
        >>> # Get September 2019 data (all days)
        >>> radiance = radiance_sept_2019_apr_2020.get('2019').get('09')
        >>> # Flatten dictionary to list of arrays
        >>> radiance_arrays = flatten_data(radiance)
        >>> # Display type
        >>> type(radiance_arrays)
        list
        >>> Display number of days
        >>> len(radiance_arrays)
        30
        # Calculate mean of arrays
        >>> radiance_mean = calculate_mean(radiance_arrays)
        # Display shape of mean array
        >>> radiance_mean.shape
        (18, 40)
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
        >>> # Define export output paths
        >>> radiance_mean_outpath = os.path.join(
        ...     output_directory,
        ...     "radiance-mean.tif")
        # Define export transform
        >>> transform = from_origin(
        ...     lon_min,
        ...     lat_max,
        ...     coord_spacing,
        ...     coord_spacing)
        >>> # Define export metadata
        >>> export_metadata = {
        ...     "driver": "GTiff",
        ...     "dtype": radiance_mean.dtype,
        ...     "nodata": 0,
        ...     "width": radiance_mean.shape[1],
        ...     "height": radiance_mean.shape[0],
        ...     "count": 1,
        ...     "crs": 'epsg:4326',
        ...     "transform": transform
        ... }
        >>> # Export mean radiance
        >>> export_array(
        >>>     array=radiance_mean,
        >>>     output_path=radiance_mean_outpath,
        >>>     metadata=export_metadata)
        Exported radiance-mean.tif
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


def subtract_arrays(minuend, subtrahend):
    """Subtract one array from another if
    the arrays have the same shape.

    Parameters
    ----------
    minuend : numpy array
        Starting array.

    subtrahend : numpy array
        Array to subtract.

    Returns
    -------
    difference : numpy arrays
        Results of the calculation: minuend - subtrahend.

    Example
    -------
        >>> # Import numpy
        >>> import numpy as np
        >>> # Create and subtract arrays
        >>> arr_1 = np.array([1, 2, 3, 4])
        >>> arr_2 = np.array([2, 2, 2, 2])
        >>> arr_diff = subtract_arrays(arr_1, arr_2)
        >>> # Display difference
        >>> array_diff
        array([-1,  0,  1,  2])
    """
    # Subtract arrays
    try:
        difference = minuend - subtrahend

    # Catch shape mismatch
    except ValueError as error:
        print(
            f'ERROR - Array shape mismatch ({minuend.shape} vs. {subtrahend.shape})\nReturning empty array.')
        difference = np.empty(0)

    # Return difference
    return difference


def extract_extent(study_area, longitude_column, latitude_column):
    """Creates a plotting extent from a
    dataframe containing pixel lat/lon values.

    Intended for use with plotting and exporting
    numpy array values, with spatial properties.

    Parameters
    ----------
    study_area : pandas dataframe
        Dataframe containing lat/lon values
        for all pixels in the study area.

    longitude_column : str
        Name of the column containing longitude
        values.

    latitude_column : str
        Name of the column containing latitude
        values.

    Returns
    ------
    extent : tuple (of float)
        Tuple (left, right, bottom, top) of the
        study area bounds.

    transform : rasterio.transform affine object
        Affine transformation for the georeferenced array.

    shape : tuple (of int)
        Shape (rows, columns) of the spatially-correct array.

    Example
    -------
        >>> # Define path to lat/lon CSV
        >>> psu_lat_lon_path = os.path.join(
        ...     radiance_directory, "PSU_Pixel_LatLongs.csv")
        >>> Read CSV into dataframe
        >>> psu_lat_lon_df = pd.read_csv(psu_lat_lon
        >>> # Create PSU extent from dataframe
        >>> psu_extent, psu_transform, psu_shape = create_plotting_extent(
        ...     study_area=psu_lat_lon_df,
        ...     longitude_column='Longitude',
        ...     latitude_column='Latitude')
        >>> # Display extent
        >>> psu_extent
        (-77.93943837333333, -77.77277170666667, 40.75700065647059, 40.83200066352941)
        >>> # Display transform
        >>> psu_transform
        Affine(0.004166666666666521, 0.0, -77.93735504,
               0.0, -0.004166667058823534, 40.82991733)
        >>> # Display shape
        >>> psu_shape
        (18, 40)
    """
    # Get number pixels in study area
    num_pixels = len(study_area.index)

    # Get number of rows in study area (unique latitude values)
    num_rows = len(study_area[latitude_column].unique())

    # Get number of columns in study area (unique longitude values)
    num_columns = len(study_area[longitude_column].unique())

    # Define array shape (rows, columns)
    shape = (num_rows, num_columns)

    # Get min/max longitude and latitude values
    longitude_min = study_area[longitude_column].min()
    longitude_max = study_area[longitude_column].max()
    latitude_min = study_area[latitude_column].min()
    latitude_max = study_area[latitude_column].max()

    # Get the spacing between rows (latitude spacing)
    row_spacing = (latitude_max - latitude_min) / (num_rows - 1)

    # Get the spacing between columns (longitude spacing)
    column_spacing = (longitude_max - longitude_min) / (num_columns - 1)

    # Define extent (lat/lon as top-left corner of pixel)
    extent = (
        longitude_min,
        longitude_max + column_spacing,
        latitude_min - row_spacing,
        latitude_max
    )

    # Define transform (top-left corner: west, north, and pixel size: xsize, ysize)
    transform = from_origin(
        longitude_min, latitude_max, column_spacing, row_spacing)

    # Return extent
    return extent, transform, shape


def create_metadata(array, transform, driver='GTiff', nodata=0, count=1, crs="epsg:4326"):
    """Creates export metadata, for use with
    exporting an array to raster format.

    Parameters
    ----------
    array : numpy array
        Array containing data for export.

    transform : rasterio.transform affine object
        Affine transformation for the georeferenced array.

    driver : str
        File type/format for export. Defaults to GeoTiff ('GTiff').

    nodata : int or float
        Value in the array indicating no data. Defaults to 0.

    count : int
        Number of bands in the array for export. Defaults to 1.

    crs : str
        Coordinate reference system for the georeferenced
        array. Defaults to EPSG 4326 ('epsg:4326').

    Returns
    -------
    metadata : dict
        Dictionary containing the export metadata.

    Example
    -------
        >>> # Imports
        >>> import numpy as np
        >>> from rasterio.transform import from_origin
        >>> # Create array
        >>> arr = np.array([[1,2],[3,4]])
        >>> transform = from_origin(-73.0, 43.0, 0.5, 0.5)
        >>> meta = create_metadata(arr, transform)
        # Display metadata
        >>> meta
        {'driver': 'GTiff',
         'dtype': dtype('int32'),
         'nodata': 0,
         'width': 2,
         'height': 2,
         'count': 1,
         'crs': 'epsg:4326',
         'transform': Affine(0.5, 0.0, -73.0,
                0.0, -0.5, 43.0)}
    """
    # Define metadata
    metadata = {
        "driver": driver,
        "dtype": array.dtype,
        "nodata": nodata,
        "width": array.shape[1],
        "height": array.shape[0],
        "count": count,
        "crs": crs,
        "transform": transform
    }

    # Return metadata
    return metadata


def store_monthly_mean(radiance_daily, dates):
    """Calculates monthly mean radiance values
    for each entry (year/month) in a list of and
    stores the monthly means in a dictionary.

    Parameters
    ----------
    radiance_daily : dict
        Dictionary containing daily radiance arrays,
        indexed by radiance['YYYY']['MM']['DD'].

    dates : list (of str)
        List containing strings of format 'YYYY-MM'.

    Returns
    -------
    radiance_monthly_mean : dict
        Dictionary containig monthly mean radiance
        arrays, indexed by radiance_monthly_mean['YYYY']['MM'].

    Example
    -------
        >>> # Define months list
        >>> months = [
        ...     '2018-09',
        ...     '2018-10',
        ...     '2018-11',
        ...     '2018-12'
        ... ]
        >>> # Store monthly means in dictionary
        >>> radiance_monthtly_mean = store_monthly_mean(
        ...     radiance_daily=radiance_sept_2018_may_2020, dates=months)
        >>> # Show top-level keys (years)
        >>> radiance_monthtly_mean.keys()
        dict_keys(['2018'])
        >>> # Show 2018 keys (months)
        >>> radiance_monthtly_mean.get('2018').keys()
        dict_keys(['09', '10', '11', '12'])
    """
    # Initialize dictionary to store monthly mean radiance arrays
    radiance_monthtly_mean = {}

    # Loop through all dates
    for date in dates:

        # Extract year and month
        year, month = date.split('-')

        # Add year to dictionary if not existing key
        if year not in radiance_monthtly_mean.keys():
            radiance_monthtly_mean[year] = {}

        # Get dictionary of daily arrays for full month
        radiance_dict = radiance_daily.get(year).get(month)

        # Flatten dictionary to list of arrays
        radiance_arrays = flatten_data(radiance_dict)

        # Calculate mean of arrays
        radiance_mean = calculate_mean(radiance_arrays)

        # Add mean array to dictionary
        radiance_monthtly_mean[year][month] = radiance_mean

    # Return monthly means
    return radiance_monthtly_mean


def store_continuous_range_mean(radiance_daily, date_range_list):
    """Calculates monthly mean radiance values
    for each entry (year/month) in a list of and
    stores the monthly means in a dictionary.

    Parameters
    ----------
    radiance_daily : dict
        Dictionary containing daily radiance arrays,
        indexed by radiance['YYYY']['MM']['DD'].

    date_ranges : list (of str)
        List containing strings of format 'YYYY-MM-DD'.

    Returns
    -------
    radiance_date_range_mean : dict
        Dictionary containig date range mean radiance arrays,
        indexed by radiance_date_range_mean['YYYYMMDD-YYYYMMDD'].

    Example
    -------
        >>> # Define date ranges
        >>> fall_2018_date_range_list = [
        ...    ('2018-09-01', '2018-12-16'),
        ...    ('2018-11-18', '2018-11-24'),
        ...    ('2018-12-08', '2018-12-14'),
        ...    ('2018-12-17', '2019-01-04'),
        ... ]
        >>> # Store means
        >>> fall_2018_means = store_continuous_range_mean(
        ...     radiance_daily=radiance_sept_2018_may_2020,
        ...     date_range_list=fall_2018_date_range_list)
        >>> # Show keys
        >>> for key in fall_2018_means.keys():
        ...     print(key)
        20180901-20181216
        20181118-20181124
        20181208-20181214
        20181217-20190104
    """
    # Create list of date ranges for start/end date combo
    date_ranges = [create_date_list(start_date, end_date)
                   for start_date, end_date in date_range_list]

    # Initialize dictionary to store monthly mean radiance arrays
    radiance_date_range_mean = {}

    # Loop through all months
    for date_range in date_ranges:

        # Create index based on date range
        date_key = f"{date_range[0].replace('-', '')}-{date_range[-1].replace('-', '')}"

        # Get array for each date into list
        radiance_arrays = extract_data(
            radiance=radiance_daily, dates=date_range)

        # Calculate mean of arrays
        radiance_mean = calculate_mean(radiance_arrays)

        # Add mean array to dictionary
        if date_key not in radiance_date_range_mean.keys():
            radiance_date_range_mean[date_key] = radiance_mean

    # Return date range means
    return radiance_date_range_mean


def store_weekly_range_mean(radiance_daily, start_date, end_date):
    """Calculates mean radiance values
    for each entry (year/month) in a list and
    stores the means in a dictionary.

    Parameters
    ----------
    radiance_daily : dict
        Dictionary containing daily radiance arrays,
        indexed by radiance['YYYY']['MM']['DD'].

    start_date : str
        String of format 'YYYY-MM-DD'.


    start_date : str
        String of format 'YYYY-MM-DD'.

    Returns
    -------
    radiance_weekly_range_mean : dict
        Dictionary containing recurring weekly mean radiance arrays,
        indexed by radiance_date_range_mean['YYYYMMDD-YYYYMMDD-DAY'].

    Example
    -------
        >>> # Store Fall 2018 data
        >>> fall_2018_weekly = store_weekly_range_mean(
        ...     radiance_daily=radiance_sept_2018_may_2020,
        ...     start_date='2018-09-01', end_date='2018-12-16')
        >>> # Display dictionary keys
        >>> for key in fall_2018_weekly.keys():
        ...     print(key)
        20180901-20181216-SUN
        20180901-20181216-MON
        20180901-20181216-TUE
        20180901-20181216-WED
        20180901-20181216-THU
        20180901-20181216-FRI
        20180901-20181216-SAT
        20180901-20181216-BUS
    """
    # Define date frequencies to loop through for creating date ranges
    day_list = ["W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "B"]

    # Create list of date ranges for each day in date list (all by default)
    date_ranges = [create_date_list(start_date, end_date, date_frequency=day)
                   for day in day_list]

    # Create string for adding to the end of the index string
    day_str = [day[-3:] if "W-" in day else f"{day}US" for day in day_list]

    # Initialize index for looping through day list
    day_index = 0

    # Initialize dictionary to store weekly range mean radiance arrays
    radiance_weekly_range_mean = {}

    # Loop through all months
    for date_range in date_ranges:

        # Create index based on date range and recurring day
        date_key = f"{start_date.replace('-', '')}-{end_date.replace('-', '')}-{day_str[day_index]}"

        # Get array for each date into list
        radiance_arrays = extract_data(
            radiance=radiance_daily, dates=date_range)

        # Calculate mean of arrays
        radiance_mean = calculate_mean(radiance_arrays)

        # Add mean array to dictionary
        if date_key not in radiance_weekly_range_mean.keys():
            radiance_weekly_range_mean[date_key] = radiance_mean

        # Add one to day index
        day_index += 1

    # Return weekly range means
    return radiance_weekly_range_mean


def unpack_dictionaries(dictionaries):
    """Flattens/unpacks a list of dictionaries into
    a single dictionary.

    Parameters
    ----------
    dictionaries : list
        List containing multiple dictionaries

    Returns
    -------
    unpacked : dict
        Dictionary containing all keys/values of
        all dictionaries in the input list.

    Example
    -------
        >>> # Define dictionaries
        >>> week_1 = {'radiance-week1': 200}
        >>> week_2 = {'radiance-week2': 300}
        >>> # Create list of dictionaries
        >>> week_list = [week_1, week_2]
        >>> week_list
        [{'radiance-week1': 200}, {'radiance-week2': 300}]
        >>> # Unpack dictionaries
        >>> unpacked = unpack_dictionaries(week_list)
        {'radiance-week1': 200, 'radiance-week2': 300}
    """
    # Reverse input list
    dictionaries_reversed = list(reversed(dictionaries))

    # Flatten/unpack all semester dictionaries into single dictionary
    unpacked = dict(ChainMap(*dictionaries_reversed))

    # Return unpacked dictionary
    return unpacked


def plot_values(radiance, location='Penn State Campus', title='Radiance', data_source='NASA Black Marble', difference=False):
    """Plots the values in a radiance array.

    Parameters
    ----------
    radiance : numpy array
        Array containing raw values, mean values,
        or difference values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    title : str, optional
        Plot sub-title. Default value is 'Radiance'.
        Intended for 'September 2019 Mean Radiance' or
        'Change in Mean Radiance (September 2019 vs.
        March 2020)'.

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    difference : bool, optional
        Boolean indicating if the array contains raw
        values or mean values (False) or contains
        difference values (True). Default value is False.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the plot.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the plot.

    Example
    -------
        >>> # Plot difference from Sept 2019 to March 2020
        >>> fig, ax = plot_values(
        ...     diff_sep_2019_march_2020,
        ...     title="Change in Mean Radiance (September 2019 vs. March 2020)",
        ...     difference=True)
    """
    # Find absolute values for radiance min & max
    radiance_min_abs = np.absolute(radiance.min())
    radiance_max_abs = np.absolute(radiance.max())

    # Determine max value (for plotting vmin/vmax)
    plot_max = radiance_min_abs if (
        radiance_min_abs > radiance_max_abs) else radiance_max_abs

    # Define vmin and vmax
    plot_vmin = -plot_max if difference else 0
    plot_vmax = plot_max

    # Define radiance units
    units = "$\mathrm{nWatts \cdot cm^{−2} \cdot sr^{−1}}$"

    # Define title
    plot_title = f"{title} ({units})"

    # Define colormap
    plot_cmap = 'RdBu_r' if difference else "Greys_r"

    # Use dark background
    with plt.style.context('dark_background'):

        # Create figure and axes object
        fig, ax = plt.subplots(figsize=(16, 8))

        # Adjust spacing
        plt.subplots_adjust(top=0.95)

        # Add super title
        plt.suptitle(f"{location} Cloud Free Radiance", size=24)

        # Plot array
        ep.plot_bands(
            radiance,
            scale=False,
            title=plot_title,
            vmin=plot_vmin,
            vmax=plot_vmax,
            cmap=plot_cmap,
            ax=ax)

        # Set title size
        ax.title.set_size(20)

        # Add caption
        fig.text(0.5, .15, f"Data Source: {data_source}",
                 ha='center', fontsize=16)

    # Return figure and axes object
    return fig, ax


def plot_histogram(radiance, location='Penn State Campus', title='Distribution of Radiance', xlabel='Radiance',
                   ylabel='Pixel Count', data_source='NASA Black Marble', difference=False):
    """Plots the distribution of values in a radiance array.

    Parameters
    ----------
    radiance : numpy array
        Array containing raw values, mean values,
        or difference values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    title : str, optional
        Plot sub-title. Default value is 'Distribution of
        Radiance'. Intended for 'Distribution of September
        2019 Mean Radiance' or 'Distribution of the Change
        in Mean Radiance (September 2019 vs. March 2020).'

    xlabel : str, optional
        Label on the x-axis. Default value is 'Radiance'.

    ylabel : str, optional
        Label on the y-axis. Default value is 'Pixel Count'.

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    difference : bool, optional
        Boolean indicating if the array contains raw
        values or mean values (False) or contains
        difference values (True). Default value is False.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the histogram.

    Example
    -------
        >>> # Plot Sept 2019 vs. March 2020 change histogram
        >>> fig, ax = plot_histogram(
        ...     diff_sep_2019_march_2020,
        ...     title="Distribution of the Change in Mean Radiance (September 2019 vs. March 2020)",
        ...     xlabel='Change in Mean Radiance',
        ...     difference=True)
    """
    # Find absolute values for radiance min & max
    radiance_min_abs = np.absolute(radiance.min())
    radiance_max_abs = np.absolute(radiance.max())

    # Determine max value (for plotting vmin/vmax)
    plot_max = radiance_min_abs if (
        radiance_min_abs > radiance_max_abs) else radiance_max_abs

    # Define vmin and vmax
    hist_min = -plot_max if difference else 0
    hist_max = plot_max

    # Define histogram range
    hist_range = (hist_min, hist_max)

    # Define radiance units
    units = "$\mathrm{nWatts \cdot cm^{−2} \cdot sr^{−1}}$"

    # Use dark background
    with plt.style.context('dark_background'):

        # Create figure and axes object
        fig, ax = ep.hist(
            radiance,
            hist_range=hist_range,
            colors='#984ea3',
            title=title,
            xlabel=f'{xlabel} ({units})',
            ylabel=ylabel)

        # Add super title
        plt.suptitle(f"{location} Cloud Free Radiance", size=24)

        # Adjust spacing
        plt.subplots_adjust(top=0.9)

        # Add caption
        fig.text(0.5, .03, f"Data Source: {data_source}",
                 ha='center', fontsize=14)

    # Return figure and axes object
    return fig, ax


def plot_change(pre_change, post_change, location='Penn State Campus',
                titles=['Radiance', 'Radiance', 'Radiance'],
                data_source='NASA Black Marble'):
    """Plots two arrays and the difference
    between the arrays on the same figure.

    pre_change : numpy array
        Numpy array containing radiance values.

    post_change : numpy array
        Numpy array containing radiance values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    titles : list of str, optional
        Plot sub-titles. Default value is ['Radiance',
        'Radiance', 'Radiance']. Intended for ['September
        2019 Mean Radiance', 'March 2020 Mean Radiance',
       'Change in Mean Radiance (September 2019 vs. March
       2020)'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>> # Define titles
        >>> plot_titles = [
        ...     'September 2019 Mean Radiance',
        ...     'March 2020 Mean Radiance',
        ...     'Change in Mean Radiance (September 2019 vs. March 2020)'
        ... ]
        >>> # Plot Sept 2019 and March 2020
        >>> fig, ax = plot_change(
        >>>     pre_change=radiance_monthtly_mean.get('2019').get('09'),
        >>>     post_change=radiance_monthtly_mean.get('2020').get('03'),
                titles=plot_titles)
    """
    # Find max radiance values in pre-change and post-change
    pre_change_max = np.absolute(pre_change.max())
    post_change_max = np.absolute(post_change.max())

    # Define vmin/vmax for radiance values
    value_vmin = 0

    value_vmax = pre_change_max if (
        pre_change_max > post_change_max) else post_change_max

    # Calculate difference (post-change - pre-change)
    diff = subtract_arrays(post_change, pre_change)

    # Find absolute values for change min & max
    diff_min_abs = np.absolute(diff.min())
    diff_max_abs = np.absolute(diff.max())

    # Determine max value (for plotting vmin/vmax)
    diff_vmax = diff_min_abs if (
        diff_min_abs > diff_max_abs) else diff_max_abs

    diff_vmin = -diff_vmax

    # Define radiance units
    units = "$\mathrm{nWatts \cdot cm^{−2} \cdot sr^{−1}}$"

    # Define titles
    pre_change_title = f"{titles[0]} ({units})"
    post_change_title = f"{titles[1]} ({units})"
    diff_title = f"{titles[2]} ({units})"

    # Define color maps
    value_cmap = 'Greys_r'
    diff_cmap = 'RdBu_r'

    # Use dark background
    with plt.style.context('dark_background'):

        # Create figure and axes objects
        fig, ax = plt.subplots(3, 1, figsize=(30, 20))

        # Add super title
        plt.suptitle(f"{location} Cloud Free Radiance", size=24)

        # Adjust spacing
        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.925)

        # Plot pre-change array
        ep.plot_bands(
            pre_change,
            scale=False,
            title=pre_change_title,
            vmin=value_vmin,
            vmax=value_vmax,
            cmap=value_cmap,
            ax=ax[0])

        # Plot post-change array
        ep.plot_bands(
            post_change,
            scale=False,
            title=post_change_title,
            vmin=value_vmin,
            vmax=value_vmax,
            cmap=value_cmap,
            ax=ax[1])

        # Plot diff array
        ep.plot_bands(
            diff,
            scale=False,
            title=diff_title,
            vmin=diff_vmin,
            vmax=diff_vmax,
            cmap=diff_cmap,
            ax=ax[2])

        # Add caption
        fig.text(0.5, .1, f"Data Source: {data_source}",
                 ha='center', fontsize=16)

        # Set title size
        ax[0].title.set_size(20)
        ax[1].title.set_size(20)
        ax[2].title.set_size(20)

    # Return figure and axes object
    return fig, ax


def save_figure(filepath):
    """Saves the current figure to a specified location.

    Parameters
    ----------
    filepath : str
        Path (including file name and extension)
        for the output file.

    Returns
    -------
    message : str
        Message indicating location of saved file
        (upon success) or error message (upon failure)

    Example
    -------
        >>> # Plot Sept 2019 mean
        >>> fig, ax = rd.plot_values(
        ...     radiance_monthtly_mean.get('2019').get('09'),
        ...     title="September 2019 Mean Radiance",
        ...     difference=False)
        >>> # Define output path
        >>> outpath = os.path.join(
        ...     working_directory,
        ...     "penn-state-campus-radiance-sept-2019.png")
        >>> # Save figure as PNG
        >>> save_figure(outpath)
        Saved plot: working_directory\\penn-state-campus-radiance-sept-2019.png
    """
    # Save figure
    try:
        plt.savefig(fname=filepath, facecolor='k',
                    dpi=300, bbox_inches="tight")

    # Set message to error
    except Exception as error:
        message = print(f"Could not save plot. ERROR: {error}")

    # Set message to output location
    else:
        message = print(f"Saved plot: {filepath}")

    # Return message
    return message
