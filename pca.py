#!usr/bin/env python  
#-*- coding: utf-8 -*-  
import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np  
import cPickle as pickle
from sklearn import metrics
from sklearn.decomposition import RandomizedPCA
import detection1
from pybrain.tools.customxml.networkwriter import NetworkWriter 

def read_data(data_file):  
    import gzip  
    f = gzip.open(data_file, "rb")  
    train, val, test = pickle.load(f)  
    f.close  
    train_x = train[0]  
    train_y = train[1]  
    test_x = test[0]  
    test_y = test[1]  
    return train_x, train_y, test_x, test_y
def pca_test(X):                               #这个函数是用来返回最佳的n值，即ng讲的那个测评函数要达到99%
    pca = RandomizedPCA()  
    pca.fit(X)
    n_components=X.shape[1]
    for n in range(10,X.shape[1],5):
      s=sum(pca.explained_variance_ratio_[:n])
      if(s>=0.99):
          n_components=n
          print n
          #print "%d is best for pca" %n_components
          break
    #pca.set_params(n_components=n_components)
    return n_components  
def pca_process(X,n_component):               #找到合适的n后训练pca模型
    pca = RandomizedPCA(n_components=n_component)  
    pca.fit(X)
    return pca
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    train_x, train_y, test_x, test_y = read_data(data_file)
    print "create pca...."
    pca = pca_process(train_x,335)
    train_z = pca.transform(train_x)
    test_z = pca.transform(test_x)            #Xtest也需要转化为Ztest
    print train_z.shape[1]
    print "create model....."
    #model = logistic_regression_classifier(train_z, train_y)
    #trainpredict = model.predict(train_z)
    #testpredict = model.predict(test_z)
    #trainaccuracy = metrics.accuracy_score(train_y, trainpredict)      
    #print 'trainaccuracy: %.2f%%' % (100 * trainaccuracy) 
    #testaccuracy = metrics.accuracy_score(test_y, testpredict)      
    #print 'testaccuracy: %.2f%%' % (100 * testaccuracy)
    start_time = time.time()
    net=detection1.create_network(train_z, train_y, test_z, test_y)
    print 'training took %fs!' % (time.time() - start_time)
    NetworkWriter.writeToFile(net, 'mynetwork2.xml')
    
