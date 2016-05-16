import sys
#print sys.path
import features as featExtract
import numpy as np
import deal_with_tif as tif
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import time
# number of features
def trainingExec(patches = [], ground = []):

    N=284
    if (len(patches) == 0 or len(ground) == 0):
        print "At least one patch is needed!"
    elif (len(patches) != len(ground)):
        print "Inconsistent lengths!"
    else:
        #print sys.path

        print 'begin the whole thing, initialize timer...'
        t0 = time.time()
        # image paths

        raw_data = '/Users/dibrov/Documents/python/runforest/FlyWingData/rawData.tif'
        ground_truth = '/Users/dibrov/Documents/python/runforest/FlyWingData/groundTruth.tif'





      #  t1 = time.time()
      #  print 'loaded training patches...%.1fs '%(t1 -t0)

        # producing features and labels for given patches

        # just p1
        # p1_features_0, p1_filter_names = fe.feat(p1)

        # N = np.shape(p1_features_0)[0]
        # p1_features = p1_features_0.flatten().reshape(N,L*W).T
        # p1_labels_0 = tif.load_crop(image_path=ground_truth, origin=(400,400), length=L, width=W)
        # p1_labels = p1_labels_0.flatten()


        # for three concatenated patches
        #features_0 = np.concatenate((fe.feat(p1),fe.feat(p2),fe.feat(p3)), axis=1)
        #features = features_0.flatten().reshape((N,3*L*W)).T
        #labels_0 = np.concatenate((tif.load_crop(image_path=ground_truth, origin=(400,400), length=L, width=W),\
        #             tif.load_crop(image_path=ground_truth, origin=(864,3032), length=L, width=W),\
        #             tif.load_crop(image_path=ground_truth, origin=(1056,1038), length=L, width=W)), axis=1)
        #
        #labels = labels_0.flatten()

        featureArraySquare = []; featureFilterNames = []

        for i in range(len(patches)):

            fe_sq_patch, fe_filter_names_patch = featExtract.feat(patches[i])
            featureArraySquare.append(fe_sq_patch); featureFilterNames.append(fe_filter_names_patch);


        fe = []
        for i in range(len(patches)):

            fe_patch = featureArraySquare[i].reshape(N,np.size(patches[i]))
            fe.append(fe_patch)


        fore_list = [] # lets select 3 4 5
        fore_list_la = []

        la = []
        for i in range(len(patches)):

            la_item = ground[i].flatten()/255
            la.append(la_item)


        features = np.concatenate(fe, axis=1).T
        labels = np.concatenate(la)

        print 'done with feature extraction for training patches... %.1f s '%(time.time() - t0)


        #-------------------------------------------------------------------------------------------#
        # getting a bigger validation set
        # vp0 = tif.load_crop(origin =(1396,2765), length=L, width=W)
        # vp0_features_square, vp0_filter_names = fe.feat(vp0)
        # vp0_features = vp0_features_square.flatten().reshape((N,L*W)).T
        # vp0_labels = tif.load_crop(image_path=ground_truth, origin=(1396,2765), length=L, width=W)

        # vp1 = tif.load_crop(origin =(1409,1290), length=L, width=W)
        # vp1_features_square, vp1_filter_names = fe.feat(vp1)
        # vp1_features = vp1_features_square.flatten().reshape((N,L*W)).T
        # vp1_labels = tif.load_crop(image_path=ground_truth, origin=(1409,1290), length=L, width=W)

        # vp2 = tif.load_crop(origin =(582,1839), length=L, width=W)
        # vp2_features_square, vp2_filter_names = fe.feat(vp2)
        # vp2_features = vp2_features_square.flatten().reshape((N,L*W)).T
        # vp2_labels = tif.load_crop(image_path=ground_truth, origin=(582,1839), length=L, width=W)


        # #-------------------------------------------------------------------------------------------#

         # Performance changes. Increasing prediction quality with higher number of max_depth (=10).

        n_estimators = [200]
        max_features = [2]
        max_depth = [10]
        #threshold = [.5]

        out = open('summary','w')
        out.write('number of trees'.rjust(15)+ 'max_features'.rjust(51)+ 'max_depth'.rjust(15)+'quant'.rjust(10)  +'\n')


        for ne in n_estimators:
            for mf in max_features:
                for md in max_depth:
                        tcl = time.time()
                        print 'begin the classification... %.1f s'%(time.time() - t0)

                        forest = RandomForestClassifier(n_estimators=ne, max_features=mf,
                                        n_jobs = 24, max_depth = md, class_weight={0:1.,1:300.})
                        forest.fit(features, labels)
                        print 'ended learning! it took %.1f s seconds '%((time.time() - tcl))

                        # validation set (one patch so far)

                        #vp0 = tif.load_crop(origin =(1396,2765), length=L, width=W)
                        #vp0_features_square, vp0_filter_names = fe.feat(vp0)
                        #vp0_features = vp0_features_square.flatten().reshape((N,L*W)).T
                        #vp0_labels = tif.load_crop(image_path=ground_truth, origin=(1396,2765), length=L, width=W)

                        #prob = forest.predict_proba(vp0_features)

                        #fore = prob[:,1].reshape((L,W))
                        #back = prob[:,0].reshape((L,W))


                       # p = forest.predict_proba(p1_features)

                        #pf = p[:,1].reshape((L,W ))


                        # ipmortance of the features
                        #dic=[]

                        #imp = forest.feature_importances_

                        #for i in range(N):
                        #    dic.append({'name':fe1_filter_names[i], 'importance':imp[i]})

                        #sort_imp = sorted(dic, key=itemgetter('importance'), reverse=True)


                        # probability prediction
                        #-------------------------------------------------------------------------------------------#
        out.write('done')
        out.close()


        outname = "rf_models/forest_24jobs.pkl"

        print "saving forest to %s " %outname


        joblib.dump(forest,outname)

