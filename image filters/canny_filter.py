from PIL import Image
from math import *
import argparse
# Run the following command
# python canny_filter.py "/Users/benjamincolebrook/Desktop/hw2_example/person.jpg" 1 2 "c_output.jpg"


# Gaussian Filter
def gaussian(x, y, pix, w, h, sigma, k):
    # Smooth image with Gaussian convolution
    rout = 0
    bout = 0
    gout = 0
    coef = 1.0/(2.0*pi*sigma)
    # for each pixel, compute the gaussian
    for m in range(max(0, x-k), min(w, x+k+1)):
        for n in range(max(0, y-k), min(h, y+k+1)):
            stuff = exp(-(((x-m)**2 + (y-n)**2)/(2.0*sigma)))
            gau = coef * stuff
            r, g, b = pix[m, n]
            rout = rout + gau * r
            gout = gout + gau * g
            bout = bout + gau * b

    return int(round(rout)), int(round(gout)), int(round(bout))


# Finds the edge strength and orientation i.e. the gradient magnitude and angle
def gradient_fns(x, y, pix, w, h):
    # Compute Jx and Jy, edge strength and edge orientation
    jxr1, jxg1, jxb1 = pix[max(0, x - 1), y]
    jxr2, jxg2, jxb2 = pix[min(w-1, x+1), y]
    jxr, jxg, jxb = jxr2 - jxr1, jxg2 - jxg1, jxb2 - jxb1
    jyr1, jyg1, jyb1 = pix[x, max(0, y-1)]
    jyr2, jyg2, jyb2 = pix[x, min(h-1, y+1)]
    jyr, jyg, jyb = jyr2 - jyr1, jyg2 - jyg1, jyb2 - jyb1
    jx = jxr + jxg + jxb
    jy = jyr + jyg + jyb
    edgstr = sqrt(jx**2 + jy**2)
    edgori = degrees(atan2(jy, jx))
    return edgstr, edgori


# Groups the pixels by orientation then suppresses th pixel to 0 if it is not greater than it's
#  neighbor in that direction
def bin_and_suppress(x, y, pix, w, h, edgstr, edgori):
    # bin
    dk = 0
    if edgori > 22.5:
        dk = 45
    if edgori > 67.5:
        dk = 90
    if edgori > 112.5:
        dk = 135
    if edgori > 157.5:
        dk = 0

    final_val = edgstr

    # suppress
    if dk == 0:
        eswest, eowest = gradient_fns(max(0, x-1), y, pix, w, h)
        eseast, eoeast = gradient_fns(min(x+1, w-1), y, pix, w, h)
        if edgstr < max(eswest, eseast):
            final_val = 0
    elif dk == 45:
        esne, eone = gradient_fns(min(x+1, w-1), min(y+1, h-1), pix, w, h)
        essw, eosw = gradient_fns(max(0, x-1), max(0, y-1), pix, w, h)
        if edgstr < max(esne, essw):
            final_val = 0
    elif dk == 90:
        esnorth, eonorth = gradient_fns(x, min(y+1, h-1), pix, w, h)
        essouth, eosouth = gradient_fns(x, max(0, y-1), pix, w, h)
        if edgstr < max(esnorth, essouth):
            final_val = 0
    elif dk == 135:
        esnw, eonw = gradient_fns(max(0, x-1), min(y+1, h-1), pix, w, h)
        esse, eose = gradient_fns(min(x+1, w-1), max(0, y-1), pix, w, h)
        if edgstr < max(esnw, esse):
            final_val = 0

    return int(round(final_val))


def main():
    # Create a python argument parser
    parser = argparse.ArgumentParser(description='code')

    # -i for the path to the input image
    parser.add_argument('-i', '--input_path', type=str, default=None,
                        help='path to image')

    # -ss for the sigma squared value in the bilateral formula
    parser.add_argument('-ss', '--sigma_squared', type=int, default=1000,
                        help='sigma d squared for bilateral formula')

    # -k for the kernel size
    parser.add_argument('-k', '--kernel', type=int, default=8,
                        help='size of kernel (for 8x8 use 8)')

    # -o for the path to the output image
    parser.add_argument('-o', '--output_path', type=str, default='resize_output.png',
                        help='output image path')
    args = parser.parse_args()

    # Open the image
    img = Image.open(args.input_path).convert("RGB")

    # Create the new image based on the width and height of the first and the final image for the edge detector
    width, height = img.size
    new_img = Image.open(args.input_path).convert("RGB")
    final_img = Image.new("L", (width, height), "black")

    # Use Pillow's load function for easier manipulation of pixels
    pixels = img.load()
    new_pixels = new_img.load()
    final_pixels = final_img.load()
    for i in range(width):
        for j in range(height):
            # For each pixel in the original image, create the new image which is essentially just the gaussian
            # smoothing similar to a bilateral filter.
            p = gaussian(i, j, pixels, width, height, args.sigma_squared, args.kernel)
            new_pixels[i, j] = p

            # Find the edge strength and orientation by using pixel gradients
            edge_strength, edge_orientation = gradient_fns(i, j, new_pixels, width, height)

            # Bin and suppress the results in the final image
            final_pix = bin_and_suppress(i, j, new_pixels, width, height, edge_strength, edge_orientation)
            final_pixels[i, j] = final_pix

    # Save the output image using Pillow
    final_img.save(args.output_path)

if __name__ == "__main__":
    main()
