from PIL import Image
import numpy as np
import argparse
from math import *


def gradient_image(img):
    """
    Find the gradient map of the image
    :param img: np array of the pixel values for the image
    :return: np array gradient map for the image
    """
    h, w = img.shape[:2]
    grad_img = np.ndarray((h, w))   # Initialize the array
    for i in range(h):
        for j in range(w):
            # Compute Jx and Jy, edge strength
            [jxr1, jxg1, jxb1] = img[i][max(0, j-1)]
            [jxr2, jxg2, jxb2] = img[i][min(w - 1, j + 1)]
            jxr, jxg, jxb = jxr2 - jxr1, jxg2 - jxg1, jxb2 - jxb1
            [jyr1, jyg1, jyb1] = img[max(0, i - 1)][j]
            [jyr2, jyg2, jyb2] = img[min(h - 1, i + 1)][j]
            jyr, jyg, jyb = jyr2 - jyr1, jyg2 - jyg1, jyb2 - jyb1
            jx = jxr + jxg + jxb
            jy = jyr + jyg + jyb
            edge_strength = sqrt(jx ** 2 + jy ** 2)
            grad_img[i, j] = edge_strength
    return grad_img


def get_seams(grad_img):
    h, w = grad_img.shape[:2]
    paths = np.zeros((h, w), dtype='int64')
    grad_totals = np.zeros((h, w), dtype='int64')
    grad_totals[0] = grad_img[0]
    for i in range(1, h):
        for j in range(w):
            prev_grads = grad_totals[i-1, max(j-1, 0): j+2]
            least_grad = prev_grads.min()
            grad_totals[i][j] = grad_img[i][j] + least_grad
            paths[i][j] = np.where(prev_grads == least_grad)[0][0] - (1*(j != 0))
    return paths, grad_totals


def seam_end(gradient_totals):
    return list(gradient_totals[-1]).index(min(gradient_totals[-1]))


def find_seam(paths, end):
    h, w = paths.shape[:2]
    seam = [end]
    for i in range(h-1, 0, -1):
        curr = seam[-1]
        prev = paths[i][curr]
        seam.append(curr + prev)
    seam.reverse()
    return seam


def remove_seam(img, seam):
    h, w = img.shape[:2]
    return np.array([np.delete(img[row], seam[row], axis=0) for row in range(h)])


def image_resize(img, target_w):
    h, w = img.shape[:2]
    for i in range(w-target_w):
        grad_img = gradient_image(img)
        paths, grad_totals = get_seams(grad_img)
        seam = find_seam(paths, seam_end(grad_totals))
        img = remove_seam(img, seam)
    return img


def main():
    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('-i', '--image_path', type=str, default=None,
                        help='path to image')
    parser.add_argument('-t', '--target_width', type=int, default=None,
                        help='target width for image')
    parser.add_argument('-o', '--output_path', type=str, default='resize_output.png')
    args = parser.parse_args()

    img = np.array(Image.open(args.image_path), dtype='int64')
    new_img = image_resize(img, args.target_width)

    Image.fromarray(new_img.astype('uint8')).save(args.output_path)


if __name__ == "__main__":
    main()

'''
python content_aware.py -i 'resize_lego.jpeg' -t 800
'''