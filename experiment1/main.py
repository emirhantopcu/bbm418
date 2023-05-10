import cv2
import numpy as np
import os
from scipy import ndimage as ndi


def edgeDetection(org_img):
    """Simple function for canny filtering

    Parameters
    ---------
    org_img :
        image to apply canny filter
    Returns
    -------
    edges_img :
        canny filter applied image
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Canny Edge Detection
    edges_img = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    return edges_img


def houghTransformOnImage(filename_str, img):
    """Simple function for canny filtering

    Parameters
    ---------
    filename_str : str
        filename of the image
    img :
        original image

    Returns
    -------
    hough_space :  3d ndarray
        hough space array
    """

    print("Currently processing file: ", filename_str)
    edges = edgeDetection(img)
    img_shape = edges.shape             # takes image as a parameter then applies edge detection
    x_max = img_shape[1]                # saves image parameters
    y_max = img_shape[0]
    radius_range = radius_range_for_samples[filename_str.strip('.jpg')]
    hough_space = []
    for radius in range(radius_range[0], radius_range[1]):      # for every radius value in the range specified in
        hough_layer = np.zeros((y_max, x_max))                  # ground truth .txt files
        print("radius = ", radius)
        for x in range(x_max):
            for y in range(y_max):
                if edges[y, x] != 0:                            # checks every pixel if its value is 0 or not, if not
                    temp_array = np.zeros((y_max, x_max))           # draws a circle for that radius value on that pixel
                    cv2.circle(temp_array, (x, y), radius, 1, 1)    # on a temporary array then sums this array with
                    hough_layer = hough_layer + temp_array          # hough layer array
        hough_space.append(hough_layer)                 # these layer arrays gets accumulated in hough space array

    hough_space = np.asarray(hough_space)
    return hough_space


def findMaxima(space):
    """Simple function for canny filtering

    Parameters
    ---------
    space :
        3d ndarray for houghspace

    Returns
    -------
    coords :
        list with coordinates of circles
    """

    img2 = ndi.maximum_filter(space, size=(5, 5, 5))    # looking in a 5x5x5 neighborhood area
    img_thresh = img2.mean() + img2.std() * 6           # setting the global threshold
    labels, num_labels = ndi.label(img2 > img_thresh)

    coords = ndi.center_of_mass(space, labels=labels, index=np.arange(1, num_labels + 1))
    return coords


def printCircles(circle_coords, img, min_radius, filename_str):
    """Simple function for canny filtering

    Parameters
    ---------
    circle_coords :
        coordinate list
    img :
        original image
    min_radius : int
        minimum radius value for that image
    filename_str : str
        filename

    Returns
    -------

    """

    for coord in circle_coords:
        # for every coordinate in the list draw a circle on the original image with desired radius, min radius value is
        # summed up with numpy array index, giving the desired radius value
        cv2.circle(img, (int(coord[2]), int(coord[1])), int(coord[0] + min_radius), (0, 0, 255), thickness=2)
        # drawing a dot for that circle indicating the center coordinate
        cv2.circle(img, (int(coord[2]), int(coord[1])), 0, (0, 0, 255), thickness=2)
    # format filepath
    savedfile = filename_str.strip('.jpg')
    savedfile = savedfile + 'Circles.png'
    # save image
    cv2.imwrite(os.path.join('results', savedfile), img)
    # print on console for tracking process
    print("Completed ", savedfile)


ground_truth = {}  # putting ground truth values in a dictionary for later use
directory_gt = 'datasetGT'
for filename in os.listdir(directory_gt):
    file = os.path.join(directory_gt, filename)
    if os.path.isfile(file):
        with open(file) as f:
            lines = f.readlines()
        no = file.split('\\')[1].strip('.txt')
        ground_truth[no] = [lines[0].strip("\n")]
        for index in range(1, len(lines)):
            ground_truth[filename.strip('.txt')] = ground_truth[filename.strip('.txt')] +\
                                                   [lines[index].strip("\n").split(" ")]


radius_range_for_samples = {}  # creating a dictionary for radius ranges, it gets calculated with the values in
for sample_number in ground_truth:  # ground truth values dictionary
    radius_list = []
    for i in range(1, len(ground_truth[sample_number])):
        radius_list.append(float(ground_truth[sample_number][i][2]))
    if not radius_list:                                         # some .txt files indicate zero circles so there is not
        radius_range_for_samples[sample_number] = (70, 110)     # a desired range, this value I set is completely
        continue                                                # arbitrary
    radius_range_for_samples[sample_number] = (int(min(radius_list)) - 5, int(max(radius_list)) + 5)

if not os.path.isdir('results'):        # if a result directory is not present create one
    os.mkdir('results')

directory_images = 'datasetIMG'         # for every file in the datasetIMG directory execute following commands
for filename_img in os.listdir(directory_images):
    file = os.path.join(directory_images, filename_img)
    circle_file_number = filename_img.strip('.jpg')
    img_original = cv2.imread(file)
    accumulator = houghTransformOnImage(filename_img, img_original)
    circle_coords_list = findMaxima(accumulator)
    printCircles(circle_coords_list, img_original, radius_range_for_samples[circle_file_number][0], filename_img)
