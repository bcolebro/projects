from PIL import Image
import numpy as np
import argparse
from math import *


def match(row_left, row_right, occ):
    """
    Implement the matching algorithm discussed in class here
    :param row_left: np array representation for a row in the left image
    :param row_right: np array representation for the corresponding row in the right image
    :param occ: int cost of occlusion
    :return:    p: 2d np array with the paths for matching rows
                c: 2d np array with the costs for each index
    """
    # Initialize the cost matrix and path matrix
    if len(row_left) != len(row_right):
        raise ValueError('Image widths do not match')

    row_len = len(row_left)
    c = np.zeros((row_len + 1, row_len + 1))
    p = np.chararray((row_len + 1, row_len + 1))

    # Set the left column and top row to initial values
    for i in range(0, row_len + 1):
        c[i][0] = i*occ
        c[0][i] = i*occ
        p[0][i] = 'r'
        p[i][0] = 'l'
    # Fill in the matrix
    for i in range(1, row_len + 1):
        for j in range(1, row_len + 1):
            occlude_left = c[i - 1][j] + occ
            occlude_right = c[i][j - 1] + occ
            match_cost = c[i - 1][j - 1] + pow(row_left[i-1] - row_right[j-1], 2)
            costs = [occlude_left, occlude_right, match_cost]
            min_cost = min(costs)
            c[i, j] = min_cost
            if min_cost == occlude_left:
                p[i][j] = 'l'
            elif min_cost == occlude_right:
                p[i][j] = 'r'
            elif min_cost == match_cost:
                p[i][j] = 'm'
    return p, c


def find_disparity(paths, l):
    """
    Given the paths found in the matching algorithm find the disparity of each row
    :param paths: 2d np array with the paths for matching rows
    :param l: length of a row (same for right and left, supposedly)
    :return: disparity of the left and right rows
    """
    disp_left = np.zeros(l)
    disp_right = np.zeros(l)
    i = l -1
    j = l -1
    while i != 0 and j != 0:
        if paths[i][j] == 'm':
            disp_left[i] = abs(i-j)
            disp_right[j] = abs(j-i)
            i = i-1
            j = j-1
        elif paths[i][j] == 'l':
            disp_left[i] = np.NaN
            i = i-1
        elif paths[i][j] == 'r':
            disp_right[j] = np.NaN
            j = j-1
    return disp_left, disp_right


def disp_image(imgl, imgr, occ):
    """
    Finds the left and right disparity images
    :param imgl: left image as a np array
    :param imgr: right image as a np array
    :param occ: occlusion cost as int
    :return: left and right disparity images
    """
    h, w = imgl.shape[:2]
    disparity_left = np.zeros((h, w), dtype='int64')
    disparity_right = np.zeros((h, w), dtype='int64')
    for row in range(h):
        paths, cost = match(imgl[row], imgr[row], occ)
        disparity_l, disparity_r = find_disparity(paths, len(imgl[row]))
        disparity_left[row] = disparity_l
        disparity_right[row] = disparity_r
    return disparity_left, disparity_right



def main():
    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('-l', '--left_image_path', type=str, default=None,
                        help='path to left image')
    parser.add_argument('-r', '--right_image_path', type=str, default=None,
                        help='path to right image')
    parser.add_argument('-p', '--left_output_path', type=str, default='left_disparity.png')
    parser.add_argument('-q', '--right_output_path', type=str, default='right_disparity.png')
    parser.add_argument('-o', '--occlusion', type=int, default=50)
    args = parser.parse_args()

    img_l = np.array(Image.open(args.left_image_path).convert("L"), dtype='int64')
    img_r = np.array(Image.open(args.right_image_path).convert("L"), dtype='int64')
    disparity_left, disparity_right = disp_image(img_l, img_r, args.occlusion)

    Image.fromarray(disparity_left.astype('uint8')).save(args.left_output_path)
    Image.fromarray(disparity_right.astype('uint8')).save(args.right_output_path)


if __name__ == "__main__":
    main()