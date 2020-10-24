import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    best_match = []
    
    for i, query_hist in enumerate(query_hists):
        min_d = np.inf
        min_d_index = -1

        for j, model_hist in enumerate(model_hists):
            d = dist_module.get_dist_by_name(query_hist, model_hist, dist_type)
            D[j][i] = d
            if d < min_d:
                min_d = d
                min_d_index = j
        best_match.append(min_d_index)

    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist

    for image_path in image_list:

        image = np.array(Image.open(image_path)).astype('double')
        if hist_isgray:
            image = rgb2gray(image)

        if hist_type == "grayvalue":
            hists, _ = histogram_module.get_hist_by_name(
                image, num_bins, hist_type)
        else:
            hists = histogram_module.get_hist_by_name(image, num_bins, hist_type)
        image_hist.append(hists)

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    _, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    D_sorted = np.argsort(D, axis=0)

    for j in range(D_sorted.shape[1]):
        img = np.array(Image.open(query_images[j]))
        plt.subplot(3, 6, j*6+1)
        plt.title("Q" + str(j))
        plt.axis("off")
        plt.imshow(img)
        i = 0
        for model_img_i in D_sorted[:num_nearest, j]:
            i += 1
            img_match = np.array(Image.open(model_images[model_img_i]))
            plt.subplot(3, 6, j*6+1+i)
            d = round(D[model_img_i, j], 2)
            plt.title("M" + str(d))
            plt.axis("off")
            plt.imshow(img_match)
    plt.show()


