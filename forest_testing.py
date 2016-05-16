import features as fe
import numpy as np
import deal_with_tif as tif
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt 
from sklearn.externals import joblib
from time import time
import sys
import scipy.misc



def time_deco(f):
    """decorates a function to print the time it took """
    def _func(*args, **kwargs):
        from time import time
        t = time()
        res = f(*args, **kwargs)
        print "time needed: %.1f seconds"%(time() - t)
        return res

    return _func


def load_forest(outname):
    """load a pickled forest outname"""
    print "loading random forest from %s"%outname 
    return joblib.load(outname)

@time_deco
def extract_features(data):
    """
    extract the features for data and reshape the output 
    to shape (N_features, N_voxels)
    """
    print "extracting features...."

    feats, names  = fe.feat(data)
    
    N_feat = len(feats)
    Ny,Nx = data.shape

    feats = feats.reshape((N_feat,Nx*Ny))
    return feats.T

@time_deco
def classify(forest, feats):
    """extract the probability map for being background (class index 1)"""

    print 'classification...'

    prob_map = forest.predict_proba(feats)[:,1]

    return prob_map



def classifyExec(path="./data/rawData.tif"):
    if not locals().has_key("forest"):
        forest = load_forest("rf_models/forest_24jobs.pkl")
    
    #wing = tif.load_raw_data()[:800,:800]
    wing = tif.load_raw_data(image_path=path)

    feats = extract_features(wing)

    res = classify(forest, feats).reshape(wing.shape)
    scipy.misc.imsave("./data/probMap.png", res, "png")


