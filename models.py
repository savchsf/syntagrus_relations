from prepare_sets import BalancedSet

#BalancedSet - data from here
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import joblib

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, f1_score, classification_report

import scalers

import shutil

import json

from glob import glob

import os

class Models:

    def __init__(self,scaler):
        self.model=None
        self.scaler=scaler

    def InitLogistic(self,max_iter=10000):
        #self.model=OneVsRestClassifier(LogisticRegression(max_iter=max_iter))
        self.model = SGDClassifier(loss='log_loss', max_iter=max_iter, class_weight='balanced')

    def InitSVM(self,kernel='rbf',C=1.0):
        self.model=OneVsRestClassifier(SVC(kernel=kernel, C=C, class_weight='balanced'))

    def train_model(self,X_train,y_train,scaled_X=True):
        if not scaled_X:
            X_train=self.scaler.fit_transform(X_train)

        self.model.fit(X_train,y_train)


    def save_model(self,path):
        joblib.dump(self.model, path)

    def load_model(self,path):
        self.model=joblib.load(path)

    def one_hot_class(self,vector,thr=0.1):
        value=np.argmax(vector)
        if vector[value]<=thr:
            return 0
        return value+1

    def evaluate_metric(self,y_test,y_pred,one_hot=True,thr=0.1):
        
        if one_hot:
            y_test=np.asarray([self.one_hot_class(x,thr) for x in y_test])
            y_pred=np.asarray([self.one_hot_class(x,thr) for x in y_pred])

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        return acc, f1, report

    def predict(self,X,scaled_X=False):
        if not scaled_X:
            X=self.scaler.transform(X)
        
        return self.model.predict(X)

class TrainModels:

    @staticmethod
    def trainmodel(b,m,load_path,save_path,head=False,thr=0.1):
        b.load_train_test(load_path)
        m.train_model(b.X_train,b.y_train)
        predictions=m.predict(b.X_test)
        acc, f1, report=m.evaluate_metric(b.y_test,predictions,not head, thr)
        m.save_model(f"{save_path}_model.pth")
        with open(save_path+'_metrics.txt','w',encoding='utf-8') as f:
            f.write(report)

    @staticmethod
    def process_one(directory,prefix,type_,save_dir,thr=0.1):
        scaler = joblib.load(directory + f'/{prefix}_scaler.pkl')
        m=Models(scaler)

        if type_=='SVM':
            m.InitSVM()
        elif type_=="Logistic":
            m.InitLogistic()
        else:
            raise NotImplementedError("Select a model (SVM or Logistic)")

        os.makedirs(save_dir,exist_ok=True)

        save_path=save_dir+'/'+prefix

        with open(f'{directory}/{prefix}.json') as f:
            scheme=json.load(f)

        b=BalancedSet()
        TrainModels.trainmodel(b,m,f'{directory}/{prefix}',save_path,head=False,thr=thr)

        if scheme['predict_head']:
            TrainModels.trainmodel(b,m,f'{directory}/{prefix}_heads',save_path+'_heads',head=True)

        shutil.copy(f'{directory}/{prefix}.json',save_path+'.json')
        shutil.copy(f'{directory}/{prefix}_bow.json',save_path+'_bow.json')

    @staticmethod
    def get_prefixes(directory):
        paths=glob(directory+'/*.json') #counting prefixes by json
        prefixes=[x.split('/')[-1].replace('.json','') for x in list(paths) if '_bow' not in x]
        return prefixes

    @staticmethod
    def process_all(directory,type_,save_dir,thr=0.1):
        prefixes=TrainModels.get_prefixes(directory)
        for prefix in prefixes:
            TrainModels.process_one(directory,prefix,type_,save_dir,thr=thr)
        

if __name__=="__main__":

    TrainModels.process_all('bigger_balanced','SVM','bigger_models',thr=0.1)
