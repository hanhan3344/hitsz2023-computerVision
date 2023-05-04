import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    end_row = start_row + num_rows
    end_col = start_col + num_cols

    out = image[start_row:end_row, start_col:end_col, :]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    for i in range(output_rows):
        for j in range(output_cols):
            # Calculate the corresponding pixel value in the input image
            input_i = round((i / output_rows) * input_rows)
            input_j = round((j / output_cols) * input_cols)
            # Copy the RGB values from the input image to the output image
            output_image[i, j, :] = input_image[input_i, input_j, :]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    x, y = point
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)
    return np.array([x_rotated, y_rotated])
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    
    # Calculate output shape
    output_rows = input_rows
    output_cols = input_cols

    # Calculate center of image
    center_x = input_cols / 2
    center_y = input_rows / 2

    # Calculate rotation matrix
    c = np.cos(-theta)
    s = np.sin(-theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    # Iterate over every pixel in the output image
    for i in range(output_rows):
        for j in range(output_cols):
            # Calculate rotated coordinates relative to center of image
            rotated_coords = rotation_matrix @ np.array([j - center_x, i - center_y])
            x = rotated_coords[0] + center_x
            y = rotated_coords[1] + center_y

            # Calculate the four pixels we need to interpolate between
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))

            # Make sure all pixels are inside the input image
            if x1 < 0 or x1 >= input_cols or y1 < 0 or y1 >= input_rows:
                continue

            # Set pixel value in output image
            output_image[i, j] = input_image[y1, x1]

    ### END YOUR CODE

    # 3. Return the output image
    return output_image
