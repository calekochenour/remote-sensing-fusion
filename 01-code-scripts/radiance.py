# Imports
import re
from collections import ChainMap
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import rasterio as rio
from rasterio.transform import from_origin
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
