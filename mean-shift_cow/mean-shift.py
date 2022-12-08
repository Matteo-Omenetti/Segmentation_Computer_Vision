import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    distance = torch.square(torch.norm(X - x, dim=1))
    return distance

def distance_batch(x, X):
    X = X.unsqueeze(1)
    X_1 = X.repeat(1, X.shape[0], 1)
    X_2 = X_1.transpose(0, 1)
    dist = torch.square(torch.norm(X_2 - X_1, dim=2))
    
    return dist

def gaussian(dist, bandwidth):
    return torch.exp(- dist / (2 * bandwidth ** 2))

def update_point(weight, X): 
    weight = weight.unsqueeze(1)
    weight_sum = torch.sum(weight)
    weight = torch.hstack([weight, weight, weight])
    return torch.sum(weight * X, dim=0) / weight_sum

def update_point_batch(weight, X):
    weight = weight.unsqueeze(2)
    # print(weight.shape)
    weight_sum = torch.sum(weight, 0)
    # print(weight_sum.shape)
    weight = weight.repeat(1, 1, 3)
    X = X.unsqueeze(1)
    X = X.repeat(1, X.shape[0], 1)
    return torch.sum(weight * X, dim=0) / weight_sum

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    dist = distance_batch(X_, X_)
    weight = gaussian(dist, bandwidth)
    X = update_point_batch(weight, X)
    
    return X
    

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)


result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
