import sys
#print sys.path
import features as fe
import numpy as np
import deal_with_tif as tif
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import time
# number of features
def trainingExec():
    N=284

    #print sys.path

    print 'begin the whole thing, initialize timer...'
    t0 = time.time()
    # image paths

    raw_data = '/Users/dibrov/Documents/python/runforest/FlyWingData/rawData.tif'
    ground_truth = '/Users/dibrov/Documents/python/runforest/FlyWingData/groundTruth.tif'

    # length and width of the patch
    L = 100
    W = 100

    # trainig set: nine pathces
    p1 = tif.load_crop(origin=(400,400), length=L, width=W)
    p2 = tif.load_crop(origin=(864,3032), length=L, width=W)
    p3 = tif.load_crop(origin=(1056,1038), length=L, width=W)
    p4 = tif.load_crop(origin=(1278,2601), length=L, width=W)
    p5 = tif.load_crop(origin=(912,2157), length=L, width=W)
    p6 = tif.load_crop(origin=(1143,1911), length=L, width=W)
    p7 = tif.load_crop(origin=(741,387), length=L, width=W)
    p8 = tif.load_crop(origin=(1548,1281), length=L, width=W)
    p9 = tif.load_crop(origin=(459,3363), length=L, width=W)

    # three more patches

    p10 = tif.load_crop(origin=(727,2262), length=L, width=W)
    p11 = tif.load_crop(origin=(944,555), length=L, width=W)
    p12 = tif.load_crop(origin=(487,660), length=L, width=W)

    # trainig set labels
    la1_sq = tif.load_crop(image_path=ground_truth, origin=(400,400), length=L, width=W)
    la2_sq = tif.load_crop(image_path=ground_truth, origin=(864,3032), length=L, width=W)
    la3_sq = tif.load_crop(image_path=ground_truth, origin=(1056,1038), length=L, width=W)
    la4_sq = tif.load_crop(image_path=ground_truth, origin=(1278,2601), length=L, width=W)
    la5_sq = tif.load_crop(image_path=ground_truth, origin=(912,2157), length=L, width=W)
    la6_sq = tif.load_crop(image_path=ground_truth, origin=(1143,1911), length=L, width=W)
    la7_sq = tif.load_crop(image_path=ground_truth, origin=(741,387), length=L, width=W)
    la8_sq = tif.load_crop(image_path=ground_truth, origin=(1548,1281), length=L, width=W)
    la9_sq = tif.load_crop(image_path=ground_truth, origin=(459,3363), length=L, width=W)

    la10_sq = tif.load_crop(image_path=ground_truth, origin=(727,2262), length=L, width=W)
    la11_sq = tif.load_crop(image_path=ground_truth, origin=(944,555), length=L, width=W)
    la12_sq = tif.load_crop(image_path=ground_truth, origin=(487,660), length=L, width=W)

    t1 = time.time()
    print 'loaded training patches...%.1fs '%(t1 -t0)

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

    fe1_sq, fe1_filter_names = fe.feat(p1)
    fe2_sq, fe2_filter_names = fe.feat(p2)
    fe3_sq, fe3_filter_names = fe.feat(p3)
    fe4_sq, fe4_filter_names = fe.feat(p4)
    fe5_sq, fe5_filter_names = fe.feat(p5)
    fe6_sq, fe6_filter_names = fe.feat(p6)
    fe7_sq, fe7_filter_names = fe.feat(p7)
    fe8_sq, fe8_filter_names = fe.feat(p8)
    fe9_sq, fe9_filter_names = fe.feat(p9)

    fe10_sq, fe10_filter_names = fe.feat(p10)
    fe11_sq, fe11_filter_names = fe.feat(p11)
    fe12_sq, fe12_filter_names = fe.feat(p12)

    fe1 = fe1_sq.reshape(N,L*W)
    fe2 = fe2_sq.reshape(N,L*W)
    fe3 = fe3_sq.reshape(N,L*W)
    fe4 = fe4_sq.reshape(N,L*W)
    fe5 = fe5_sq.reshape(N,L*W)
    fe6 = fe6_sq.reshape(N,L*W)
    fe7 = fe7_sq.reshape(N,L*W)
    fe8 = fe8_sq.reshape(N,L*W)
    fe9 = fe9_sq.reshape(N,L*W)

    fe10 = fe10_sq.reshape(N,L*W)
    fe11 = fe11_sq.reshape(N,L*W)
    fe12 = fe12_sq.reshape(N,L*W)

    fore_list = [] # lets select 3 4 5
    fore_list_la = []



    la1 = la1_sq.flatten()/255
    la2 = la2_sq.flatten()/255
    la3 = la3_sq.flatten()/255
    la4 = la4_sq.flatten()/255
    la5 = la5_sq.flatten()/255
    la6 = la6_sq.flatten()/255
    la7 = la7_sq.flatten()/255
    la8 = la8_sq.flatten()/255
    la9 = la9_sq.flatten()/255

    la10 = la10_sq.flatten()/255
    la11 = la11_sq.flatten()/255
    la12 = la12_sq.flatten()/255

    for j in range(np.shape(fe1)[1]):
        if la3[j]==1.0:
                fore_list.append(fe3[:,j])
                fore_list_la.append(la3[j])

        if la4[j]==1.0:
                fore_list.append(fe4[:,j])
                fore_list_la.append(la4[j])
        if la5[j]==1.0:
                fore_list.append(fe5[:,j])
                fore_list_la.append(la5[j])

    fl = np.asarray(fore_list)
    fll = np.asarray(fore_list_la)

    features = np.concatenate((fe1,fe2,fe6,fe7,fe8,fe9,fe10,fe11,fe12,fl.T), axis=1).T
    labels = np.concatenate((la1,la2,la6,la7,la8,la9,la10,la11,la12,fll))

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

