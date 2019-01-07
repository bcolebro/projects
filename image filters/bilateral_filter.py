from math import *
from PIL import Image
import argparse

'''
This program takes an image as input and saves another image as output that has the bilateral filter applied
to it. This is essentially a smoothing filter. You can run this program with a command similar to:

python bilateral_filter.py -i aaron_rogers.jpg -sd 2000 -sr 2 -o aaron_rogers_bf.png

In this directory, there is an image of Aaron Rogers named aaron_rogers.jpg. The image with the bilateral filter
is also in this directory and is named aaron_rogers_bf.png
'''


# This function computes the bilateral filter formula for pixel i, j
def computation(i, j, pix, w, h, sigr, sigd, k):
    # Initialize local variables
    s = k/2
    r1, g1, b1 = pix[i, j]
    rout = 0
    gout = 0
    bout = 0
    wsum = 0

    # For each pixel in the kernel, apply the filter to get the average
    for k in range(max(0, i-s), min(w, i+s+1)):
        for l in range(max(0, j-s), min(h, j+s+1)):
            r2, g2, b2 = pix[k, l]
            d = exp(-((i-k)**2+(j-l)**2)/(2*sigr))
            r = exp(-((r1-r2)**2+(b1-b2)**2+(g1-g2)**2)/(2*sigd))
            w = d*r
            wsum = wsum + w
            rout = rout + r2*w
            gout = gout + g2*w
            bout = bout + b2*w

    # Return the new pixels
    rout = int(round(rout/wsum))
    gout = int(round(gout/wsum))
    bout = int(round(bout/wsum))
    return rout, gout, bout


def main():
    # Create a python argument parser
    parser = argparse.ArgumentParser(description='code')

    # -i for the path to the input image
    parser.add_argument('-i', '--input_path', type=str, default=None,
                        help='path to image')

    # -sd for the sigma_d squared value in the bilateral formula
    parser.add_argument('-sd', '--sigma_d_squared', type=int, default=1000,
                        help='sigma d squared for bilateral formula')

    # -sr for the sigma_r squared value in the bilateral formula
    parser.add_argument('-sr', '--sigma_r_squared', type=int, default=2,
                        help='sigma r squared for bilateral formula')

    # -k for the kernel size
    parser.add_argument('-k', '--kernel', type=int, default=8,
                        help='size of kernel (for 8x8 use 8)')

    # -o for the path to the output image
    parser.add_argument('-o', '--output_path', type=str, default='resize_output.png',
                        help='output image path')
    args = parser.parse_args()

    # Open the input image using Pillow
    img = Image.open(args.input_path).convert("RGB")

    # Create the new image based on the width and height of the first
    width, height = img.size
    new_img = Image.open(args.input_path).convert("RGB")

    # Use Pillow's load function for easier manipulation of pixels
    pixels = img.load()
    new_pixels = new_img.load()
    for i in range(width):
        for j in range(height):
            # For each pixel in the original image, use the bilateral filter formula to compute the value for the
            # pixel in the new image.
            p = computation(i, j, pixels, width, height, args.sigma_r_squared, args.sigma_d_squared, args.kernel)
            new_pixels[i, j] = p

    # Save the new image to the output path using Pillow
    new_img.save(args.output_path)


if __name__ == "__main__":
    main()

