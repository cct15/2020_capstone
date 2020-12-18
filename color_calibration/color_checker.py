import cv2
import numpy as np
from collections import namedtuple

from colour.models import cctf_decoding, cctf_encoding
from colour.utilities import as_float_array, as_int_array, as_int

ASPECT_RATIO = 1.5
"""
Colour checker aspect ratio.
"""

SWATCHES_HORIZONTAL = 6
"""
Colour checker horizontal swatches count.
"""

SWATCHES_VERTICAL = 4
"""
Colour checker vertical swatches count.
"""

SWATCHES = SWATCHES_HORIZONTAL * SWATCHES_VERTICAL
"""
Colour checker total swatches count.
"""

SWATCH_MINIMUM_AREA_FACTOR = 10
"""
Swatch minimum area factor :math:`f` with the minimum area :math:`m_a`
expressed as follows: :math:`m_a = image_w * image_h / s_c / f` where
:math:`image_w`, :math:`image_h` and :math:`s_c` are respectively the image
width, height and the swatches count.
"""

WORKING_WIDTH = 1440
"""
Width processed images are resized to.
"""

def as_8_bit_BGR_image(image):
    """
    Converts and encodes given linear float *RGB* image to 8-bit *BGR* with
    *sRGB* reverse OETF.

    Parameters
    ----------
    image : array_like
        Image to convert.

    Returns
    -------
    ndarray
        Converted image.
    """

    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    return cv2.cvtColor((cctf_encoding(image) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def adjust_image(image, target_width=WORKING_WIDTH):
    """
    Adjusts given image so that it is horizontal and resizes it to given target
    width.

    Parameters
    ----------
    image : array_like
        Image to adjust.
    target_width : int, optional
        Width the image is resized to.

    Returns
    -------
    ndarray
        Resized image.
    """

    width, height = image.shape[1], image.shape[0]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    ratio = width / target_width

    if np.allclose(ratio, 1):
        return image
    else:
        return cv2.resize(
            image, (as_int(target_width), as_int(height / ratio)),
            interpolation=cv2.INTER_CUBIC)


def is_square(contour, tolerance=0.2): # half square meets tolerance = 0.2
    """
    Returns if given contour is a square.

    Parameters
    ----------
    contour : array_like
        Shape to test whether it is a square.
    tolerance : numeric, optional
        Tolerance under which the contour is considered to be a square.

    Returns
    -------
    bool
        Whether given contour is a square.

    """

    return cv2.matchShapes(contour, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                           cv2.CONTOURS_MATCH_I2, 0.0) < tolerance


def contour_centroid(contour):
    """
    Returns the centroid of given contour.

    Parameters
    ----------
    contour : array_like
        Contour to return the centroid of.

    Returns
    -------
    tuple
        Contour centroid.

    Notes
    -----
    -   A :class:`tuple` class is returned instead of a :class:`ndarray` class
        for convenience with *OpenCV*.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> contour_centroid(contour)
    (0.5, 0.5)
    """

    moments = cv2.moments(contour)
    centroid = np.array(
        [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])

    return centroid[0], centroid[1]


def scale_contour(contour, factor):
    """
    Scales given contour by given scale factor.

    Parameters
    ----------
    contour : array_like
        Contour to scale.
    factor : numeric
        Scale factor.

    Returns
    -------
    ndarray
        Scaled contour.

    Examples
    --------
    >>> contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> scale_contour(contour, 2)
    array([[ 0.,  0.],
           [ 2.,  0.],
           [ 2.,  2.],
           [ 0.,  2.]])
    """

    centroid = as_int_array(contour_centroid(contour))
    scaled_contour = (as_float_array(contour) - centroid) * factor + centroid

    return scaled_contour


def crop_and_level_image_with_rectangle(image, rectangle):
    """
    Crops and rotates/levels given image using given rectangle.

    Parameters
    ----------
    image : array_like
        Image to crop and rotate/level.
    rectangle : tuple
        Rectangle used to crop and rotate/level the image.

    Returns
    -------
    ndarray
        Cropped and rotated/levelled image.

    References
    ----------
    :cite:`Abecassis2011`

    Notes
    -----
    -   ``image`` is expected to be an unsigned 8-bit sRGB encoded image.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
    >>> image = as_8_bit_BGR_image(adjust_image(read_image(path)))
    >>> rectangle = (
    ...     (723.29608154, 465.50939941),
    ...     (461.24377441, 696.34759522),
    ...     -88.18692780,
    ... )
    >>> print(image.shape)
    (958, 1440, 3)
    >>> image = crop_and_level_image_with_rectangle(image, rectangle)
    >>> print(image.shape)
    (461, 696, 3)
    """

    width, height = image.shape[1], image.shape[0]
    width_r, height_r = rectangle[1]
    centroid = as_int_array(contour_centroid(cv2.boxPoints(rectangle)))
    centroid = centroid[0], centroid[1]
    angle = rectangle[-1]

    if angle < -45:
        angle += 90
        width_r, height_r = height_r, width_r

    width_r, height_r = as_int_array([width_r, height_r])

    M_r = cv2.getRotationMatrix2D(centroid, angle, 1)

    image_r = cv2.warpAffine(image, M_r, (width, height), cv2.INTER_CUBIC)
    image_c = cv2.getRectSubPix(image_r, (width_r, height_r),
                                (centroid[0], centroid[1]))

    return image_c

def colour_checkers_segmentation(image, additional_data=False):
    """
    Detects the colour checkers coordinates in given image :math:`image` using
    segmentation.

    This is the core detection definition. The process is a follows:

    -   Input image :math:`image` is converted to a grayscale image
        :math:`image_g`.
    -   Image :math:`image_g` is denoised.
    -   Image :math:`image_g` is thresholded/segmented to image
        :math:`image_s`.
    -   Image :math:`image_s` is eroded and dilated to cleanup remaining noise.
    -   Contours are detected on image :math:`image_s`.
    -   Contours are filtered to only keep squares/swatches above and below
        defined surface area.
    -   Squares/swatches are clustered to isolate region-of-interest that are
        potentially colour checkers: Contours are scaled by a third so that
        colour checkers swatches are expected to be joined, creating a large
        rectangular cluster. Rectangles are fitted to the clusters.
    -   Clusters with an aspect ratio different to the expected one are
        rejected, a side-effect is that the complementary pane of the
        *X-Rite* *ColorChecker Passport* is omitted.
    -   Clusters with a number of swatches close to :attr:`SWATCHES` are
        kept.

    Parameters
    ----------
    image : array_like
        Image to detect the colour checkers in.
    additional_data : bool, optional
        Whether to output additional data.

    Returns
    -------
    list or ColourCheckersDetectionData
        List of colour checkers coordinates or
        :class:`ColourCheckersDetectionData` class instance with additional
        data.

    Notes
    -----
    -   Multiple colour checkers can be detected if presented in ``image``.

    Examples
    --------
    >>> import os
    >>> from colour import read_image
    >>> from colour_checker_detection import TESTS_RESOURCES_DIRECTORY
    >>> path = os.path.join(TESTS_RESOURCES_DIRECTORY,
    ...                     'colour_checker_detection', 'detection',
    ...                     'IMG_1967.png')
    >>> image = read_image(path)
    >>> colour_checkers_coordinates_segmentation(image)
    [array([[1065,  707],
           [ 369,  688],
           [ 382,  226],
           [1078,  246]])]
    """

    image = as_8_bit_BGR_image(adjust_image(image, WORKING_WIDTH))

    width, height = image.shape[1], image.shape[0]
    maximum_area = width * height / SWATCHES
    minimum_area = width * height / SWATCHES / SWATCH_MINIMUM_AREA_FACTOR

    block_size = as_int(WORKING_WIDTH * 0.015)
    block_size = block_size - block_size % 2 + 1

    # Thresholding/Segmentation.
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image_g = cv2.fastNlMeansDenoising(image_g, None, 10, 7, 21)
    image_s = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, block_size, 3)
    # Cleanup.
    kernel = np.ones((3, 3), np.uint8)
    image_c = cv2.erode(image_s, kernel, iterations=1)
    image_c = cv2.dilate(image_c, kernel, iterations=1)

    # Detecting contours.
    contours, _hierarchy = cv2.findContours(image_c, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)

    # Filtering squares/swatches contours.
    swatches = []
    for contour in contours:
        curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),
                                 True)
        if minimum_area < cv2.contourArea(curve) < maximum_area and is_square(
                curve):
            swatches.append(
                as_int_array(cv2.boxPoints(cv2.minAreaRect(curve))))

    # Clustering squares/swatches.
    clusters = np.zeros(image.shape, dtype=np.uint8)
    for swatch in [
            as_int_array(scale_contour(swatch, 1 + 1 / 3))
            for swatch in swatches
    ]:
        cv2.drawContours(clusters, [swatch], -1, [255] * 3, -1)
    clusters = cv2.cvtColor(clusters, cv2.COLOR_RGB2GRAY)
    clusters, _hierarchy = cv2.findContours(clusters, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    clusters = [
        as_int_array(
            scale_contour(cv2.boxPoints(cv2.minAreaRect(cluster)), 0.975))
        for cluster in clusters
    ]

    # Filtering clusters using their aspect ratio.
    filtered_clusters = []
    for cluster in clusters[:]:
        rectangle = cv2.minAreaRect(cluster)
        width = max(rectangle[1][0], rectangle[1][1])
        height = min(rectangle[1][0], rectangle[1][1])
        ratio = width / height
        if ASPECT_RATIO * 5 /6 < ratio < ASPECT_RATIO * 4 / 3:
            filtered_clusters.append(cluster)
    clusters = filtered_clusters

    # Filtering swatches within cluster.
    counts = []
    for cluster in clusters:
        count = 0
        for swatch in swatches:
            if cv2.pointPolygonTest(cluster, contour_centroid(swatch),
                                    False) == 1:
                count += 1
        counts.append(count)
    counts = np.array(counts)
    indexes = np.where(
        np.logical_and(counts >= SWATCHES * 0.5,
                       counts <= SWATCHES * 1.25))[0].tolist()

    
    # Get the coordinates of the color checker
    if len(indexes)==0: return [[(1,1,1)]], None, None
    else:
        if len(indexes)>1: print('More than one cards are detected!!')
        colour_checkers = [clusters[i] for i in indexes]

        #???Filter the color checker
        colour_checker = colour_checkers[0]

        #Get the coordinates of the swatch
        block = []
        for sw in swatches:
          if cv2.pointPolygonTest(colour_checkers[0], contour_centroid(sw),False) == 1:
            block.append(sw)


        #Crop the color checker
        colour_checker = crop_and_level_image_with_rectangle(
            image, cv2.minAreaRect(colour_checker))
        width, height = (colour_checker.shape[1], colour_checker.shape[0])

        if width < height:
            colour_checker = cv2.rotate(colour_checker,
                                        cv2.ROTATE_90_CLOCKWISE)


        if additional_data: return colour_checker, swatches, clusters[0]
        else: return colour_checker, swatches, clusters[0]