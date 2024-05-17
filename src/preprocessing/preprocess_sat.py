""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""

import sys

import numpy as np
import xarray as xr
import scipy.ndimage

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


def gaussian_filter(data_array: xr.DataArray, sigma: float = 1) -> xr.DataArray:
    """
    For each date and band in the data_array, apply a gaussian filter with the given sigma.
    The gaussian filter should be applied to each (height, width) image individually using the
    scipy.ndimage.gaussian_filter function that has been imported.

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be filtered. The shape of the array is (date, band, height, width).
    sigma : float
        The sigma of the gaussian filter.
    Returns
    -------
    xr.DataArray
        The filtered data_array. The shape of the array is (date, band, height, width).
    """

    filtered_data = data_array.copy(deep=True)

    # iterate by date
    for date in range(data_array.shape[0]):
    
        # iterate by band
        for band in range(data_array.shape[1]):
        
            # apply the scipy.ndimage gaussian_filter function to the data_array[date][band]
            # and pass sigma
            filtered_data[date][band] = scipy.ndimage.gaussian_filter(
                data_array[date][band], sigma
            )
            

    # return the filtered_data as a xr.DataArray
    return filtered_data

def quantile_clip(
    data_array: xr.DataArray, clip_quantile: float, group_by_time=True
) -> xr.DataArray:
    """
    This function clips the outliers of the data_array by the given clip_quantile.
    It calculates the q1 and q2 using the np.quantile function. q1 and q2 are calculated
    with the method="higher" and method="lower" parameters, this clips (np.clip) any value
    above the top to the first value under the top value and any value below the bottom
    to the first value above the bottom value.

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be clipped. The shape of the array is (date, band, height, width).
    clip_quantile : float
        The quantile to clip the outliers by. Value between 0 and 0.5.
    group_by_time : bool
        affects how q1 and q2 are calculated
            if group_by_time is true: The quantile limits are shared along the time dimension (:, band).
            if group_by_time is false: The quantile limits are calculated individually for each (date, band).
    Returns
    -------
    xr.DataArray
        The clipped image data_array. The shape of the array is (date, band, height, width).
    """

    clipped_data = data_array.copy(deep=True)

    # if group_by_time is true
    if group_by_time:
    
        # iterate by band
        for band in range(data_array.shape[1]):
        
            # calculate the q1 and q2 for the band using the higher and lower methods
            q1 = np.quantile(data_array[:, band], clip_quantile, method="higher")
            q2 = np.quantile(data_array[:, band], 1 - clip_quantile, method="lower")
            
            # iterate by date
            for date in range(data_array.shape[0]):
            
                # clip the image (date, band) using q1 and q2
                clipped_data[date, band] = np.clip(data_array[date, band], q1, q2)
            
    # else
    else:
    
        # iterate by band
        for band in range(data_array.shape[1]):
        
            # iterate by date
            for date in range(data_array.shape[0]):
            
                # get quantile for the image (date, band) using the higher and lower methods
                q1 = np.quantile(data_array[date, band], clip_quantile, method="higher")
                q2 = np.quantile(data_array[date, band], 1 - clip_quantile, method="lower")
                
                # apply clipping for each image (date, band) using q1 and q2
                clipped_data[date, band] = np.clip(data_array[date, band], q1, q2)
                

    # return the data_array
    return clipped_data


def minmax_scale(data_array: xr.DataArray, group_by_time=True) -> xr.DataArray:
    """
    This function minmax scales the data_array to values between 0 and 1.
    This transforms any image to have a range between img_min to img_max
    to an image with the range [0, 1], using the formula
    (pixel_value - img_min)/(img_max - img_min).

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be minmax scaled. The shape of the array is (date, band, height, width).
    group_by_time : bool
        affects how minmax_scale operates
            if group_by_time is true: The min and max are shared along the time dimension(:, band).
            if group_by_time is false: The min and max are calculated individually for each image (date, band).
    Returns
    -------
    xr.DataArray
        The minmax scaled data_array. The shape of the array is (date, band, height, width).
    """

    scaled_data = data_array.copy(deep=True)

    # if group_by_time is true
    if group_by_time:
    
        # iterate by band
        for band in range(data_array.shape[1]):
        
            # create 2 lists to store the minimums and maximums of the images
            min_list = []
            max_list = []
            
            # iterate by date
            for date in range(data_array.shape[0]):
            
                # append the minimum value and maximum value of the image (date, band)
                min_list.append(np.min(data_array[date, band].values))
                max_list.append(np.max(data_array[date, band].values))
               

            # get the min of the minimum list and the max of the maximum list
            img_min = np.min(min_list)
            img_max = np.max(max_list)
            

            # iterate by date
            for date in range(data_array.shape[0]):
            
                # if the min == max, set the image (date, band) to an array of ones using
                # np.ones. Sometimes, satellite images can just be blank, so we would either make
                # the image all 0's or all 1's, and because we do not want any divide by zero errors
                # down the road, we will set it all to an array of ones.
                if img_min == img_max:
                
                    # set the array to np.ones
                    scaled_data[date, band] = np.ones(data_array.shape[2:])
                    
                # else
                else:
                    # use the (pixel_value - img_min)/(img_max - img_min) formula to calculate the scaled
                    # image (date, band). If you use np functions to do the math, you can replace the pixel_value
                    # with the image (date, band) and it will calculate it as a matrix operation.
                    scaled_data[date, band] = (data_array[date, band] - img_min) / (img_max - img_min)
                    
    # else
    else:
    
        # iterate by date
        for date in range(data_array.shape[0]):
        
            # iterate by band
            for band in range(data_array.shape[1]):
            
                # calculate the min and max of the image (date, band)
                img_min = np.min(data_array[date, band].values)
                img_max = np.max(data_array[date, band].values)

                # img_min = np.min([img_min])
                # img_max = np.max([img_max])
               

                # if the min == max, set the image (date, band) to an array of ones using
                # np.ones. Sometimes, satellite images can just be blank, so we would either make
                # the image all 0's or all 1's, and because we do not want any divide by zero errors
                # down the road, we will set it all to an array of ones.
                if img_min == img_max:
                
                    # set the array to np.ones
                    scaled_data[date, band] = np.ones(data_array.shape[2:])
                
                # else
                else:
                
                    # use the (pixel_value - img_min)/(img_max - img_min) formula to calculate the scaled
                    # image (date, band). If you use np functions to do the math, you can replace the pixel_value
                    # with the image (date, band) and it will calculate it as a matrix operation.
                    scaled_data[date, band] = (data_array[date, band] - img_min) / (img_max - img_min)
                    
    # return the data_array
    return scaled_data


def brighten(
    data_array: xr.DataArray, alpha: float = 0.13, beta: float = 0
) -> xr.DataArray:
    """
    Brightens the image using the formula (alpha * pixel_value + beta).

    ----------
    data_array : xr.DataArray
        The data_array to be brightened. The shape of the array is (date, band, height, width).
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.
    Returns
    -------
    xr.DataArray
        The brightened image. The shape of the array is (date, band, height, width).
    """

    brightened_data = data_array.copy(deep=True)

    # iterate by date
    for date in range(data_array.shape[0]):
    
        # iterate by band
        for band in range(data_array.shape[1]):
        
            # brighten the image using the formula. If you use np functions to do the math, you can replace the pixel_value
            # with the image (date, band) and it will calculate it as a matrix operation.
            brightened_data[date, band] = alpha * data_array[date, band] + beta
            
    # return the data_array
    return brightened_data


def gammacorr(data_array: xr.DataArray, gamma: float = 2) -> xr.DataArray:
    """
    This function applies a gamma correction to the image using the
    formula (pixel_value ^ (1/gamma))

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be brightened. The shape of the array is (date, band, height, width).
    gamma : float
        The gamma parameter of the gamma correction.
    Returns
    -------
    xr.DataArray
        The gamma corrected image. The shape of the array is (date, band, height, width).
    """

    gamma_data = data_array.copy(deep=True)

    # iterate by date
    for date in range(data_array.shape[0]):
    
        # iterate by band
        for band in range(data_array.shape[1]):
       
            # gamma correct the image using the formula. If you use np functions to do the math, you can replace the pixel_value
            # with the image (date, band) and it will calculate it as a matrix operation.
            gamma_data[date, band] = data_array[date, band] ** (1 / gamma)
            
    # return the data_array
    return gamma_data


def convert_data_to_db(data_array: xr.DataArray) -> xr.DataArray:
    """
    This function converts raw Sentinel-1 SAR data to decibel (dB) format
    using a logarithmic transformation.

    SAR (Synthetic Aperture Radar) data collected by Sentinel-1 satellites
    is initially recorded in digital numbers, representing the strength of
    the radar signal received by the satellite. The raw digital numbers
    do not provide an intuitive measure of signal intensity and can be
    affected by various factors such as system gain and antenna characteristics.

    1. Standardization: Expressing the data in decibels provides a standardized
       scale for signal intensity, making it easier to compare and interpret
       the intensity of radar returns across different areas and images.

    2. Dynamic range compression: The logarithmic scale of decibels compresses
       the dynamic range of the data, enhancing the visibility of weaker
       signals and reducing the dominance of strong signals. This is particularly
       useful for visualizing and analyzing SAR data, where the range of signal
       intensities can be very large.

    3. Interpretability: Decibel values provide a more intuitive measure of
       signal strength, with higher values indicating stronger signals and
       lower values indicating weaker signals. This facilitates interpretation
       of SAR imagery and enables users to identify features and patterns
       within the data more easily.

    By converting to decibel format it improves interpretation and analysis.

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be brightened. The shape of the array is (date, band, height, width).
    gamma : float
        The gamma parameter of the gamma correction.
    Returns
    -------
    xr.DataArray
        The gamma corrected image. The shape of the array is (date, band, height, width).
    """

    # create a new data array (db_data) with the same shape as the data_array
    db_data = data_array.copy(deep=True)

    # convert the data_array.values to log using np.log10, making sure np.where(array, np.nan, array)
    # is used to replace any 0's with np.nan before taking the log10
    db_data.values = np.log10(np.where(data_array.values == 0, np.nan, data_array.values))
   
    # return the data_array
    return db_data


def maxprojection_viirs(data_array: xr.DataArray) -> xr.DataArray:
    """
    This function takes a VIIRS data_array and returns a single image that is the max projection of the images
    to identify areas with the highest levels of nighttime lights or electricity usage.

    The value of any pixel is the maximum value over all time steps, like shown below

       Date 1               Date 2                      Output
    -------------       -------------               -------------
    | 0 | 1 | 0 |       | 2 | 0 | 0 |               | 2 | 1 | 0 |
    -------------       -------------   ======>     -------------
    | 0 | 0 | 3 |       | 0 | 4 | 0 |   ======>     | 0 | 4 | 3 |
    -------------       -------------   ======>     -------------
    | 9 | 6 | 0 |       | 0 | 8 | 7 |               | 9 | 8 | 7 |
    -------------       -------------               -------------

    Parameters
    ----------
    data_array : xr.DataArray
        The data_array to be brightened. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        Max projection of the VIIRS data_array. The shape of the array is (date, band, height, width)
    """


    # set the band index to 0 (VIIRS only has 1 band)
    band = 0
   
    # set the maximum to the first image (date, band) from the data_array
    max_viirs_array = data_array[0, band].to_numpy()

    # iterate by date
    for date in range(1, data_array.shape[0]):

        # set the maximum to be the max of (maximum, current image (date, band)),
        # this can be done numerous ways, here are some suggestions:
        # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        max_viirs_array = np.maximum(max_viirs_array, data_array[date, band].to_numpy())


    # create a new data array (max_viirs_array) with shape (1, 1, 800, 800) and the
    # relevant dims and coords. You can use np.reshape to transform the maximum (you
    # just calculated it above) to have shape (1, 1, 800, 800)
    max_viirs_array = xr.DataArray(
        max_viirs_array.reshape(1, 1, 800, 800),
        dims=["date", "band", "height", "width"],
        coords={"date": [0], "band": [0]},
    )
    

    # set the attributes of the max_viirs_array. The satellite_type should be the
    # SatelliteType.VIIRS_MAX_PROJ, while the other 2 attributes can be the same as the
    # original data_array
    max_viirs_array.attrs = data_array.attrs
    max_viirs_array.attrs["satellite_type"] = SatelliteType.VIIRS_MAX_PROJ
   
    # return the max_viirs_array
    return max_viirs_array


def preprocess_sentinel1(
    sentinel1_data_array: xr.DataArray, clip_quantile: float = 0.01, sigma=1
) -> xr.DataArray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers
        - Apply a gaussian filter
        - Minmax scale

    Parameters
    ----------
    sentinel1_data_array : xr.DataArray
        The sentinel1_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed sentinel1_data_array. The shape of the array is (date, band, height, width).
    """
    # convert data to db
    db_data = convert_data_to_db(sentinel1_data_array)
   
    # quantile clip (make sure to pass the parameter)
    quantile_clip_data = quantile_clip(db_data, clip_quantile)
    
    # apply a gaussian_filter (make sure to pass the parameter)
    gaussian_filtered_data = gaussian_filter(quantile_clip_data, sigma)
    
    # minmax and return
    return minmax_scale(gaussian_filtered_data)


def preprocess_sentinel2(
    sentinel2_data_array: xr.DataArray, clip_quantile: float = 0.05, gamma: float = 2.2
) -> xr.DataArray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers
        - Apply a gamma correction
        - Minmax scale

    Parameters
    ----------
    sentinel2_data_array : xr.DataArray
        The sentinel2_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed sentinel2_data_array. The shape of the array is (date, band, height, width).
    """
    # quantile clip (make sure to pass the parameter)
    quantile_clip_data = quantile_clip(sentinel2_data_array, clip_quantile)
    
    # gamma correct (make sure to pass the parameter)
    gamma_corrected_data = gammacorr(quantile_clip_data, gamma)
  
    # minmax and return
    return minmax_scale(gamma_corrected_data)


def preprocess_landsat(
    landsat_data_array: xr.DataArray, clip_quantile: float = 0.01, gamma: float = 2.2
) -> xr.DataArray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers
        - Apply a gamma correction
        - Minmax scale

    Parameters
    ----------
    landsat_data_array : xr.DataArray
        The landsat_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed landsat_data_array. The shape of the array is (date, band, height, width).
    """
    # quantile clip (make sure to pass the parameter)
    quantile_clip_data = quantile_clip(landsat_data_array, clip_quantile)
    
    # gamma correct (make sure to pass the parameter)
    gamma_corrected_data = gammacorr(quantile_clip_data, gamma)
    
    # minmax and return
    return minmax_scale(gamma_corrected_data)


def preprocess_viirs(
    viirs_data_array: xr.DataArray, clip_quantile: float = 0.05
) -> xr.DataArray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers
        - Minmax scale

    Parameters
    ----------
    viirs_data_array : xr.DataArray
        The viirs_data_array to be preprocessed. The shape of the array is (date, band, height, width).
    Returns
    -------
    xr.DataArray
        The processed viirs_data_array. The shape of the array is (date, band, height, width).
    """
    # quantile clip (make sure to pass the parameter)
    quantile_clip_data = quantile_clip(viirs_data_array, clip_quantile)
    
    # minmax and return
    return minmax_scale(quantile_clip_data)


if __name__ == "__main__":
    

    # test minmax_scale

    blank_data_array = xr.DataArray(
        data=np.ones(shape=(2, 2, 5, 5)),
        dims=("date", "band", "height", "width"),
        coords={
            "date": 4,
            "band": 3,
            "height": 800,
            "width": 800,
        },
    )
    blank_data_array.attrs["satellite_type"] = SatelliteType.LANDSAT
    blank_data_array.attrs["tile_dir"] = "tile_1"
    blank_data_array.attrs["parent_tile_id"] = "tile_1"

    blank_data_array.values = np.zeros(
            blank_data_array.shape
        ) + np.multiply(np.random.rand(*blank_data_array.shape), 10)
    
    data_array = minmax_scale(blank_data_array, group_by_time=True)

    print(data_array)
