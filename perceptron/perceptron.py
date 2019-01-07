from PIL import Image
from math import *
import pickle
import random
import os

m = .05  # margin
e = 0.1  # eta
iterations = 5  # number of times the perceptron runs on the weights


# Below are the HOG functions

# Center and crop image
def center_crop(image, n_width, n_height):
    width, height = image.size
    left = ceil((width - n_width)/2.)
    right = ceil((width + n_width)/2.)
    top = ceil((height - n_height)/2.)
    bottom = ceil((height + n_height)/2.)
    c_image = image.crop((left, top, right, bottom))
    return c_image


# Calculate the image gradients
def gradients(x, y, pixels, w, h):
    gx = (pixels[min(w - 1, x + 1), y] - pixels[max(0, x - 1), y]) / 2
    gy = (pixels[x, min(h - 1, y + 1)] - pixels[x, max(0, y - 1)]) / 2
    g_mag = sqrt(gx ** 2 + gy ** 2)
    g_dir = degrees(atan2(gy, gx))
    return g_mag, g_dir


# Compute histogram of gradients for 8x8 cell
def hog8x8(left, top, pixels, w, h):
    # create a key, value list for the binned histogram
    hog = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(left, left + 8):
        for j in range(top, top + 8):
            p_mag, p_dir = gradients(i, j, pixels, w, h)
            p_dir = fabs(p_dir)
            if (p_dir % 20) != 0:
                bl = p_dir - (p_dir % 20)
                br = p_dir + 20 - (p_dir % 20)
                pl = 1 - ((p_dir % 20) / 20.0)
                pr = (p_dir % 20) / 20.0
                bli = int(round(bl / 20.0))
                bri = int(round(br / 20.0))
                hog[bli] = hog[bli] + pl * p_mag
                hog[bri] = hog[bri] + pr * p_mag
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
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


# Find final vector to return
def hog_calculator(in_file):
    # Load the image using Pillow
    im = Image.open(in_file).convert("L")

    # Crop and resize
    img = center_crop(im, 64, 128)
    width, height = img.size
    pixels = img.load()

    final_vec = []
    # For each 16x16 block
    for i in range(width / 8 - 1):
        for j in range(height / 8 - 1):
            hog1 = hog8x8(int(round(float(i) * 8)), int(round(float(j) * 8)), pixels, width, height)
            hog2 = hog8x8(int(round((float(i) * 8) + 8)), int(round(float(j) * 8)), pixels, width, height)
            hog3 = hog8x8(int(round(float(i) * 8)), int(round((float(j) * 8) + 8)), pixels, width, height)
            hog4 = hog8x8(int(round((float(i) * 8) + 8)), int(round((float(j) * 8) + 8)), pixels, width, height)
            hog = hog1 + hog2 + hog3 + hog4
            norm16x16 = norm(hog)
            norm_hog = [safe_div(x, norm16x16) for x in hog]
            final_vec = final_vec + norm_hog
    # For debugging purposes
    if len(final_vec) != 3780:
        print(in_file + " not 3780 but " + str(len(final_vec)))
    return final_vec


# Find the dot product of two vectors
def dot(a, b):
    sum_ = 0
    for (i, j) in zip(a, b):
        sum_ = sum_ + i*j
    return sum_


# Run the perceptron algorithm to find the weight vector
def run_epoch(w, xs, labels, margin, eta):
    for (x, y) in zip(xs, labels):
        guard = y*dot(w, x)
        if guard < margin:
            for i in range(len(w)):
                w[i] = w[i] + eta*y*x[i]
    return w


def pickle_training_lists():
    xs = []
    labels = []

    n_list = open("train/neg.lst", "r")
    p_list = open("train/pos.lst", "r")

    for n_file in n_list:
        xs.append(hog_calculator(n_file.strip()))
        labels.append(-1)
    for p_file in p_list:
        xs.append(hog_calculator(p_file.strip()))
        labels.append(1)

    pickle.dump(xs, open("xs_train.p", "wb"))
    pickle.dump(labels, open("labels_train.p", "wb"))


def shuffle_lists(xs, labels):
    indices = range(len(xs))
    random.shuffle(indices)
    xs_ = []
    labels_ = []
    for i in indices:
        xs_.append(xs[i])
        labels_.append(labels[i])
    return xs_, labels_


def unpickle_lists():
    xs = pickle.load(open("xs_train.p", "rb"))
    labels = pickle.load(open("labels_train.p", "rb"))
    return xs, labels


def main():
    # Step 1: If the hog descriptors have not been found, find and pickle them
    if not (os.path.isfile('xs_train.p') and (os.path.isfile('labels_train.p'))):
        pickle_training_lists()

    # Step 2: Unpickle the hog descriptors and labels
    xs, labels = unpickle_lists()

    for i in range(iterations):
        # Step 3: shuffle the descriptors and labels
        xs_shuf, labels_shuf = shuffle_lists(xs, labels)

        # Step 4: Run the perceptron algorithm. If a weight does not exist, use the 0 vector else use the existing weights
        if not (os.path.isfile('save_w.p')):
            w = [0.0] * len(xs[0])
        else:
            w = pickle.load(open("save_w.p", "rb"))

        # Step 5: Train the weights on only a portion of the training data
        length = len(xs)
        xs_f = xs_shuf[0:int(round(length*0.75))]
        labels_f = labels_shuf[0:int(round(length*0.75))]

        w = run_epoch(w, xs_f, labels_f, m, e)

        # Save the new weight vector as a pickle file
        pickle.dump(w, open("save_w.p", "wb"))

        print("Iteration " + str(i + 1) + " complete.")

    print("Weights trained " + str(iterations) + " times with the perceptron.")


if __name__ == "__main__":
    main()
