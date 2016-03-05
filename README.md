# Basic-machine-learning-algorithms
以mnist手写数字当训练集，使用skit-learn和pybrain工具包，使用各种基础算法进行训练


2层网络，每层50个节点，没使用pca，784个feature

Total error: 0.0178439026584
train finish....
epoch:    1   train error: 14.27%   test error: 14.06%
Total error: 0.0167024054303
train finish....
epoch:    2   train error: 18.28%   test error: 17.16%
Total error: 0.0165783690138
train finish....
epoch:    3   train error: 17.36%   test error: 16.51%
Total error: 0.0164617111447
train finish....
epoch:    4   train error: 20.66%   test error: 19.93%
Total error: 0.016393732596
train finish....
epoch:    5   train error: 16.77%   test error: 16.42%
Total error: 0.0163370619902
train finish....
epoch:    6   train error: 18.55%   test error: 17.92%
Total error: 0.0162927244274
train finish....
epoch:    7   train error: 19.58%   test error: 18.57%
Total error: 0.0163183323357
train finish....
epoch:    8   train error: 18.79%   test error: 17.92%
Total error: 0.0162744132748
train finish....
epoch:    9   train error: 17.82%   test error: 16.77%
Total error: 0.0162878695222
train finish....
epoch:   10   train error: 15.91%   test error: 14.82%
Total error: 0.0162705544584
train finish....
epoch:   11   train error: 17.17%   test error: 16.14%
Total error: 0.0162794698131
train finish....
epoch:   12   train error: 14.75%   test error: 14.16%
Total error: 0.0162610944208
train finish....
epoch:   13   train error: 14.93%   test error: 14.35%
13
Total error: 0.0162327032017
train finish....
epoch:   14   train error: 16.63%   test error: 16.00%
Total error: 0.0162563105768
train finish....
epoch:   15   train error: 16.13%   test error: 15.73%
Total error: 0.0162445140063
train finish....
epoch:   16   train error: 15.46%   test error: 15.06%
Total error: 0.0162577410881
train finish....
epoch:   17   train error: 19.85%   test error: 18.76%
Total error: 0.0162493883265
train finish....
epoch:   18   train error: 17.72%   test error: 16.86%
Total error: 0.0162738523541
train finish....
epoch:   19   train error: 16.48%   test error: 16.04%
Total error: 0.016272623887
train finish....
epoch:   20   train error: 17.70%   test error: 17.17%

其他机器学习算法

reading training and testing data...
******************** Data Info *********************
#training data: 50000, #testing_data: 10000, dimension: 784
******************* NB ********************
training took 6.919778s!
accuracy: 83.69%
******************* KNN ********************
training took 27.188612s!
accuracy: 96.64%
******************* LR ********************
training took 109.885828s!
accuracy: 91.98%
******************* RF ********************
training took 7.944588s!
accuracy: 94.01%
******************* DT ********************
training took 48.240907s!
accuracy: 87.04%
******************* SVM ********************
training took 6038.711516s!
accuracy: 94.35%
******************* GBDT ********************
training took 7360.714812s!
accuracy: 96.17%


两层网络，使用pca后335个feature，每层60个节点

create pca....
n_compontes = 335
create model.....
335
Total error: 0.0188069646553
train finish....
epoch:    1   train error: 16.13%   test error: 15.92%
1
Total error: 0.0170109501122
train finish....
epoch:    2   train error: 16.36%   test error: 15.66%
2
Total error: 0.0167530350852
train finish....
epoch:    3   train error: 17.80%   test error: 17.06%
3
Total error: 0.0166757490514
train finish....
epoch:    4   train error: 17.03%   test error: 16.16%
4
Total error: 0.016650202723
train finish....
epoch:    5   train error: 17.48%   test error: 16.97%
5
Total error: 0.0166019048484
train finish....
epoch:    6   train error: 17.29%   test error: 16.49%
6
Total error: 0.0165702847632
train finish....
epoch:    7   train error: 18.14%   test error: 16.97%
7
Total error: 0.0165095100433
train finish....
epoch:    8   train error: 15.64%   test error: 15.09%
8
Total error: 0.0164988330919
train finish....
epoch:    9   train error: 15.99%   test error: 15.51%
9
Total error: 0.0165114423688
train finish....
epoch:   10   train error: 19.30%   test error: 18.77%
10
Total error: 0.0164744782284
train finish....
epoch:   11   train error: 17.88%   test error: 16.88%
11
Total error: 0.0164676615739
train finish....
epoch:   12   train error: 16.49%   test error: 15.82%
12
Total error: 0.0164493295057
train finish....
epoch:   13   train error: 17.78%   test error: 16.79%
13
Total error: 0.0164211944941
train finish....
epoch:   14   train error: 16.39%   test error: 15.67%
14
Total error: 0.0164268378456
train finish....
epoch:   15   train error: 16.79%   test error: 16.12%
15
Total error: 0.0164005613509
train finish....
epoch:   16   train error: 15.39%   test error: 14.84%
16
Total error: 0.0163958385784
train finish....
epoch:   17   train error: 15.98%   test error: 15.58%
17
Total error: 0.0163698472244
train finish....
epoch:   18   train error: 17.42%   test error: 17.00%
18
Total error: 0.01639310093
train finish....
epoch:   19   train error: 16.24%   test error: 15.35%
19
Total error: 0.0163815263524
train finish....
epoch:   20   train error: 14.95%   test error: 14.38%
training took 2591.329314s!
