from prepare_sets import BalancedSet

#BalancedSet - data from here
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import joblib

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, f1_score, classification_report

import scalers

class Models:

    def __init__(self,scaler):
        self.model=None
        self.scaler=scaler

    def InitLogistic(self,max_iter=10000):
        #self.model=OneVsRestClassifier(LogisticRegression(max_iter=max_iter))
        self.model = SGDClassifier(loss='log_loss', max_iter=max_iter)

    def InitSVM(self,kernel='rbf',C=1.0):
        self.model=OneVsRestClassifier(SVC(kernel=kernel, C=C))

    def train_model(self,X_train,y_train,scaled_X=True):
        if not scaled_X:
            X_train=self.scaler.fit_transform(X_train)

        self.model.fit(X_train,y_train)


    def save_model(self,path):
        joblib.dump(self.model, path)

    def load_model(self,path):
        self.model=joblib.load(path)

    def evaluate_metric(self,y_test,y_pred):
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        return acc, f1, report

    def predict(self,X,scaled_X=False):
        if not scaled_X:
            X=self.scaler.transform(X)
        
        return self.model.predict(X)

if __name__=="__main__":

    folder='bigger'
    prefix='0'

    scaler = joblib.load(folder + f'/{prefix}_scaler.pkl')
    m=Models(scaler)
    m.InitSVM()

    b=BalancedSet(scaler)
    b.load_train_test(f'{folder}/{prefix}') #not heads

    '''m.train_model(b.X_train,b.y_train)
    predictions=m.predict(b.X_test)
    acc, f1, report=m.evaluate_metric(b.y_test,predictions)
    m.save_model(f"{folder}/{prefix}_model.pth")

    print(report)

    b.load_train_test(f'{folder}/{prefix}_heads') #heads
    m.train_model(b.X_train,b.y_train)
    predictions=m.predict(b.X_test)
    acc, f1, report=m.evaluate_metric(b.y_test,predictions)
    print(report)
    m.save_model(f"{folder}/{prefix}_heads_model.pth")'''

    b.load_train_test(f'{folder}/{prefix}_heads')
    m.load_model(folder+f'/{prefix}_heads_model.pth')
    predictions=m.predict(b.X_test)
    acc, f1, report=m.evaluate_metric(b.y_test,predictions)
    print(report)