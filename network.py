#!usr/bin/env python  
#-*- coding: utf-8 -*-  
import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np  
import cPickle as pickle
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.unsupervised.trainers.deepbelief import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
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

def create_network(X,Y,testx,testy):
    numOfFeature=X.shape[1]
    numOfExample=X.shape[0]
    alldata = ClassificationDataSet(numOfFeature, 1, nb_classes=10)   #创建分类数据组
    for i in range(0,numOfExample):
        alldata.addSample(X[i], Y[i])
    alldata._convertToOneOfMany()

    numOfFeature1=testx.shape[1]
    numOfExample1=testx.shape[0]
    testdata = ClassificationDataSet(numOfFeature1, 1, nb_classes=10)   #创建分类数据组
    for i in range(0,numOfExample1):
        testdata.addSample(testx[i],testy[i])
    testdata._convertToOneOfMany()

    print alldata.indim
    print alldata.outdim
    net = FeedForwardNetwork()
    inLayer = LinearLayer(alldata.indim)
    hiddenLayer1 = SigmoidLayer(60)      #层数自己定，但是从训练效果来看，并不是网络层数和节点数越多越好
    hiddenLayer2 = SigmoidLayer(60) 
    outLayer = SoftmaxLayer(alldata.outdim)
    #bias = BiasUnit('bias')
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer1)
    net.addModule(hiddenLayer2)
    net.addOutputModule(outLayer)
    #net.addModule(bias)
    in_to_hidden = FullConnection(inLayer, hiddenLayer1)
    hidden_to_out = FullConnection(hiddenLayer2, outLayer)
    hidden_to_hidden = FullConnection(hiddenLayer1, hiddenLayer2)
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_hidden)
    net.addConnection(hidden_to_out)
    net.sortModules()

    #fnn = buildNetwork( alldata.indim, 100, alldata.outdim, outclass=SoftmaxLayer )
    trainer = BackpropTrainer( net, dataset=alldata, momentum=0.1, verbose=True, weightdecay=0.01)
    for i in range(0,20):
        print i
        trainer.trainEpochs( 1 )     #将数据训练一次
        print "train finish...."
        outtrain = net.activateOnDataset(alldata)   
        outtrain = outtrain.argmax(axis=1)  # the highest output activation gives the class，每个样本取最大概率的类 out=[[1],[2],[3],[2]...]
        outtest = net.activateOnDataset(testdata)   
        outtest = outtest.argmax(axis=1)  # the highest output activation gives the class，每个样本取最大概率的类 out=[[1],[2],[3],[2]...]
        trnresult = percentError( outtrain,alldata['class'] )
        tstresult = percentError( outtest,testdata['class'] )
        #trnresult = percentError( trainer.testOnClassData(dataset=alldata),alldata['class'] )
        #tstresult = percentError( trainer.testOnClassData(dataset=testdata),testdata['class'] )
        print "epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,"  test error: %5.2f%%" % tstresult

    return net

def load_network(fName):
    net = NetworkReader.readFrom('mynetwork1.xml')
    return net
   
if __name__ == '__main__':  
    data_file = "mnist.pkl.gz"  
    print '******************* FFN ********************' 
    train_x, train_y, test_x, test_y = read_data(data_file)
    start_time = time.time()
    net=create_network(train_x, train_y, test_x, test_y)
    print 'training took %fs!' % (time.time() - start_time)
    NetworkWriter.writeToFile(net, 'mynetwork1.xml')
