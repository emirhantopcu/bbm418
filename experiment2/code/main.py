import os
import random

import numpy as np
import cv2
from matplotlib import pyplot as plt


def findMatches(img1, img2):
    # finding matches with orb detector
    surf_detector = cv2.ORB_create()
    kp1f, des1 = surf_detector.detectAndCompute(img1, None)
    kp2f, des2 = surf_detector.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1f, img2, kp2f, matches, None, flags=2)
    cv2.imshow("img3", img3)
    cv2.waitKey(0)
    return matches, kp1f, kp2f


def findHomography(samples, keypoints_img1, keypoints_img2):
    # finding homography matrix for given sample of matches
    a_matrix = []
    # put every match in 'A' matrix in correct format
    for match in samples:
        img1_idx_h = match.queryIdx
        img2_idx_h = match.trainIdx

        source_point = keypoints_img1[img1_idx_h].pt
        destination_point = keypoints_img2[img2_idx_h].pt
        row1 = [source_point[0], source_point[1], 1.0, 0.0, 0.0, 0.0,
                -(destination_point[0] * source_point[0]),
                -(destination_point[0] * source_point[1]),
                -(destination_point[0])]
        row2 = [0.0, 0.0, 0.0, source_point[0], source_point[1], 1.0,
                -(destination_point[1] * source_point[0]),
                -(destination_point[1] * source_point[1]),
                -(destination_point[1])]
        a_matrix.append(row1)
        a_matrix.append(row2)
    a_matrix = np.matrix(a_matrix)
    u, s, vh = np.linalg.svd(a_matrix)  # numpy function for solving eigen problem for A * A' matrices
    homography = np.reshape(vh[8], (3, 3))  # rearranging the smallest eigenvector into a matrix
    homography = homography/homography[2, 2]  # dividing the matrix by its right bottom value to make it 1
    return homography


def RANSACforOptimalHomography(matches, kp1, kp2):
    # ransac algorithm for finding opimal homography matrix
    inlier_error = 1
    max_inliers = 0
    optimal_homography = None
    for i in range(1000):
        print("Max inliers = ", max_inliers)
        four_random_matches = random.sample(matches, 4)  # take 4 random samples in A matrix and calculate homography
        homography = findHomography(four_random_matches, kp1, kp2)
        inlier_count = 0
        for match in matches:  # try this homography for every match in the matches list
            img1_index = match.queryIdx
            img2_index = match.trainIdx

            source_point_ransac = kp1[img1_index].pt
            destination_point_ransac = kp2[img2_index].pt

            source_point_matrix = np.asarray([source_point_ransac[0], source_point_ransac[1], 1.0]).T
            estimated_point = np.dot(homography, source_point_matrix)
            estimated_point_x = estimated_point[0, 0]
            estimated_point_y = estimated_point[0, 1]
            estimated_point_array = np.array((estimated_point_x, estimated_point_y))
            destination_point_array = np.array(destination_point_ransac)
            # estimated point is the one calculated with the homography
            # calculate euclidian distance between estimated point and destination point of the match
            geometric_distance = np.linalg.norm(destination_point_array - estimated_point_array)
            if geometric_distance < inlier_error:
                # if the geometric distance between two points is neglectable, increase inlier count
                inlier_count = inlier_count + 1
        # if inlier count for current homography matrix is the greatest that homography should be the optimal one
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            optimal_homography = homography
    return optimal_homography


def dotHomoghraphy(h_m, x_coord, y_coord):
    # calculating estimated point with homography matrix
    point_matrix = np.asarray([x_coord, y_coord, 1.0]).T
    estimated_point = np.dot(h_m, point_matrix)
    estimated_point = estimated_point/estimated_point[0, 2]
    ep_x = estimated_point[0, 0]
    ep_y = estimated_point[0, 1]
    return ep_x, ep_y


def paste_slices(tup):              # functions for pasting an array into a bigger array
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos+w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w-max(pos+w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]


def inShape(x_coord, y_coord, shape_of_image):
    # to check if the estimated point is in range of dimensions of the canvas
    if y_coord in range(shape_of_image[0]) and x_coord in range(shape_of_image[1]):
        return True
    return False


def stitchImages(first_image, second_image):
    m, kp1, kp2 = findMatches(second_image, first_image)
    h = RANSACforOptimalHomography(m, kp1, kp2)

    image1_shape_y = first_image.shape[0]
    image1_shape_x = first_image.shape[1]

    image2_shape_y = second_image.shape[0]
    image2_shape_x = second_image.shape[1]
    # calculating the sufficient canvas size for stitching
    dr_x, dr_y = dotHomoghraphy(h, image2_shape_x, image2_shape_y)
    ur_x, ur_y = dotHomoghraphy(h, image2_shape_x, 0)
    # creating a blank canvas and pasting the foundation image on it
    shape = (image1_shape_y, int(max(dr_x, ur_x)))
    canvas = np.zeros(shape, dtype=int)
    paste(canvas, first_image, (0, 0))
    # now we have a canvas with foundation image on left

    # to warp the second image and stitch it on the canvas
    for y in range(image2_shape_y):
        for x in range(image2_shape_x):
            # iterate through each pixel on the image that will be stitched
            new_coord_x, new_coord_y = dotHomoghraphy(h, x, y)
            # calculate the coordinate of the pixel on the canvas with homography matrix
            if inShape(int(new_coord_x), int(new_coord_y), shape):
                # if found pixel is in the dimensions of the canvas print it
                canvas[int(new_coord_y), int(new_coord_x)] = second_image[y, x]

    return canvas.astype(np.uint8)


directory_name = "cvc01passadis-cyl-pano01"

# inital images before the loop
img1 = cv2.imread(os.path.join(directory_name, os.listdir(directory_name)[6]), cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread(os.path.join(directory_name, os.listdir(directory_name)[7]), cv2.IMREAD_GRAYSCALE)

# stitch them together and use the resulting image in the loop A
current_canvas = stitchImages(img1, img2)

cv2.imshow("current_canvas", current_canvas)
cv2.waitKey(0)

for i in range(8, 33):
    current_image = cv2.imread(os.path.join(directory_name, os.listdir(directory_name)[i]), cv2.IMREAD_GRAYSCALE)
    current_canvas = stitchImages(current_canvas, current_image)
    cv2.imshow("asdasd", current_canvas)
    cv2.waitKey(0)
    print("Current iteration = ", i)


plt.gray()
plt.imshow(current_canvas)

plt.show()



