from prepare_sets import BalancedSet

#BalancedSet - data from here
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import joblib

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import scalers

class Models:

    def __init__(self,scaler):
        self.model=None
        self.scaler=scaler

    def InitLogistic(self,max_iter=10000):
        self.model=OneVsRestClassifier(LogisticRegression(max_iter=max_iter))

    def InitSVM(self,kernel='rbf',C=1.0):
        self.model=OneVsRestClassifier(SVC(kernel=kernel, C=C))

    def scale_X(self,X):
        for i in range(len(X)):
            self.scaler.fit(X[i])
            X[i]=self.scaler.transform(X[i])
        return X

    def train_model(self,X_train,y_train,scaled_X=True):
        if not scaled_X:
            X_train=self.scale_X(X_train)

        self.model.fit(X_train,y_train)

    def save_model(self,path):
        joblib.dump(self.model, path)

    def load_model(self,path):
        self.model=joblib.load(path)

    def evaluate_metric(self,y_test,y_pred):
        mse_r=mean_squared_error(y_test, y_pred)
        r2_r=r2_score(y_test, y_pred)
        return mse_r,r2_r

    def predict(self,X,scaled_X=False):
        if not scaled_X:
            X=self.scale_X(X)
        
        return self.model.predict(X)

if __name__=="__main__":
    scaler=scalers.MinMaxScaler()
    m=Models(scaler)
    m.InitSVM()

    b=BalancedSet()
    b.load_train_test('slice/train_test_0.pt')

    m.train_model(b.X_train,b.y_train)
    predictions=m.predict(b.X_test)
    mse_r,r2_r=m.evaluate_metric(b.y_test,predictions)

    print(mse_r,r2_r)
