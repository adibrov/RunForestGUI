from PIL import Image
import numpy as np

def tif2array(image_path):
    ## Imports an arbitrary .tif image and creates a numpy array with the pixel-values
    im = Image.open(image_path)
    return np.array(im.getdata()).reshape(im.size[::-1])
    
def load_ground_truth(image_path = "./data/groundTruth.tif"):
    ## Imports a .tif image with the ground truth (drosophila wing) 
    ## and creates a numpy array with the pixel-values
    
    im = Image.open(image_path)
    return np.array(im.getdata()).reshape(im.size[::-1])
    
def load_raw_data(image_path = "./data/rawData.tif"):
    ## Imports a .tif image with the raw data (drosophila wing)
    ## and creates a numpy array with the pixel-values
    im = Image.open(image_path)
    return np.array(im.getdata()).reshape(im.size[::-1])
    
def load_crop(image_path="./data/rawData.tif", origin=(400,400), length=100, width=100):
    ## Imports an arbitrary .tif image region
    ## and creates a numpy array with the pixel-values

    im = Image.open(image_path)
    return np.array(im.getdata()).reshape(im.size[::-1])[origin[0]:origin[0]+length,origin[1]:origin[1]+width]
