from PIL import Image
from math import *
import sys


# Center and crop image
def center_crop(image, n_width, n_height):
    width, height = image.size
    left = ceil((width - n_width)/2.)
    right = floor((width + n_width)/2.)
    top = ceil((height - n_height)/2.)
    bottom = floor((height + n_height)/2.)
    c_image = image.crop((left, top, right, bottom))
    return c_image


# Calculate the image gradients
def gradients(x, y, pixels, w, h):
    gx = (pixels[min(w-1, x+1), y] - pixels[max(0, x-1), y])/2
    gy = (pixels[x, min(h-1, y+1)] - pixels[x, max(0, y-1)])/2
    g_mag = sqrt(gx**2 + gy**2)
    g_dir = degrees(atan2(gy, gx))
    return g_mag, g_dir


# compute histogram of gradients for 8x8 cell
def hog8x8(left, top, pixels, w, h):
    # create a key, value list for the binned histogram
    hog = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(left, left + 8):
        for j in range(top, top + 8):
            p_mag, p_dir = gradients(i, j, pixels, w, h)
            p_dir = fabs(p_dir)
            if(p_dir % 20) != 0:
                bl = p_dir - (p_dir % 20)
                br = p_dir + 20 - (p_dir % 20)
                pl = 1 - ((p_dir % 20) / 20.0)
                pr = (p_dir % 20) / 20.0
                bli = int(round(bl / 20.0))
                bri = int(round(br / 20.0))
                hog[bli] = hog[bli] + pl*p_mag
                hog[bri] = hog[bri] + pr*p_mag
            else:
                b = p_dir
                bi = int(round(b / 20.0))
                hog[bi] = hog[bi] + p_mag
    hog[0] += hog[9]
    return hog[:9]


# Compute the norm of a vector
def norm(vector):
    sum_v = 0
    for v in vector:
        sum_v = sum_v + v**2
    return sqrt(sum_v)


# Compute safe division
def safe_div(x, y):
    if y == 0:
        return 0
    return x / y


# Find final vector to return
def hog_calculator(img):
    width, height = img.size
    pixels = img.load()

    final_vec = []
    # For each 16x16 block
    for i in range(width/8 - 1):
        for j in range(height/8 - 1):
            hog1 = hog8x8(int(round(float(i)*8)), int(round(float(j)*8)), pixels, width, height)
            hog2 = hog8x8(int(round((float(i)*8) + 8)), int(round(float(j)*8)), pixels, width, height)
            hog3 = hog8x8(int(round(float(i)*8)), int(round((float(j)*8) + 8)), pixels, width, height)
            hog4 = hog8x8(int(round((float(i)*8) + 8)), int(round((float(j)*8) + 8)), pixels, width, height)
            hog = hog1+hog2+hog3+hog4
            norm16x16 = norm(hog)
            norm_hog = [safe_div(x, norm16x16) for x in hog]
            final_vec = final_vec + norm_hog
    return final_vec


def main():
    # Use the argument file as the image
    in_file = sys.argv[1]  # python hog.py "img1.png"

    # Load the image using Pillow
    im = Image.open(in_file).convert("L")

    # Crop and resize
    img = center_crop(im, 64, 128)

    # Print the hog vector
    print(len(hog_calculator(img)))


if __name__ == "__main__":
    main()
