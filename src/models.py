
#imports
from __future__ import division
import os, sys
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from keras.models import *
from keras.layers import Dense, Activation
from keras.optimizers import *
import threading
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class LogModel(threading.Thread):
    
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        for i in range(9):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(9):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())
        logModel = LogisticRegression(C=10000)
        logModel.fit(X, Y)
        sd = logModel.predict(XT)
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Logistic Regression Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of Logistic Regression Model: %.2f" % (acc)+' %')



class SVMModel(threading.Thread):
    
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        for i in range(9):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(9):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())
        svModel = SVC(kernel='rbf')
        svModel.fit(X, Y)
        sd = svModel.predict(XT)
        acc =  (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of SVM Model: %.2f"%acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of SVM Model: %.2f" % (acc)+' %')


class DTModel(threading.Thread):
    
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        dtModel = DecisionTreeClassifier()
        dtModel.fit(X, Y)
        sd = dtModel.predict(XT)
        ## StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label
        # 2011/08/10 09:46:59.607825,1.026539,tcp,94.44.127.113,1577,   ->,147.32.84.59,6881,S_RA,0,0,4,276,156,flow=Background-Established-         cmpgw-CVUT
        #([float(dur), protoDict[proto], int(Sport), int(Dport), Sip, Dip, int(totP), int(totB), stateDict[state]])
        new_input = [[1.026539,0,1577,6881,94.44,147.32,4,276,61105]]
        sd2 = dtModel.predict(new_input)
        print(sd2)
        new_input2 = [[3514.08,0,1039,65520,147.32,60.190,120,7767,2989]]
        sd3 = dtModel.predict(new_input2)
        print(sd3)
        print(X.shape)
        print(XT.shape)
        print(Y.shape)
        print(YT.shape)
        print(dtModel)
        print('=' * 100)
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Decision Tree Model: %.2f" % acc+' %')
        print('=' * 100)
        print(confusion_matrix(YT, sd))
        #print(classification_report(YT, sd))
        if self.accLabel: self.accLabel.set("Accuracy of Decision Tree Model: %.2f" % (acc)+' %')


class NBModel(threading.Thread):
    
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        nbModel = GaussianNB()
        nbModel.fit(X, Y)
        sd = nbModel.predict(XT)
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Gaussian Naive Bayes Model: %.2f" % acc +' %')
        print('='*100)
        if self.accLabel: self.accLabel.set("Accuracy of Gaussian Naive Bayes Model: %.2f" % (acc)+' %')


class KNNModel(threading.Thread):
    
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        for i in range(9):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(9):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())
        knnModel = KNeighborsClassifier()
        knnModel.fit(X, Y)
        sd = knnModel.predict(XT)
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of KNN Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of KNN Model: %.2f" % (acc)+' %')


class ANNModel(threading.Thread):
    
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        # X = self.X
        # Y = self.Y
        # XT = self.XT
        # YT = self.YT
        for i in range(9):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(9):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())

        model = Sequential()
        model.add(Dense(10, input_dim=9, activation="sigmoid"))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(1))
        sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                  loss='mse')
        model.fit(X, Y, nb_epoch=200, batch_size=100)
        sd = model.predict(XT)
        sd = sd[:, 0]
        sdList = []
        for z in sd:
            if z>=0.5:
                sdList.append(1)
            else:
                sdList.append(0)
        sdList = np.array(sdList)
        acc = (sum(sdList == YT) / len(YT) * 100)
        print("Accuracy of ANN Model: %.2f" % acc+" %")
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of ANN Model: %.2f" % (acc)+" %")
