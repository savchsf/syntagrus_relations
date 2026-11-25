from prepare_sets import BalancedSet

#BalancedSet - data from here
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

import joblib

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, f1_score, classification_report

import scalers

import shutil

import json

from glob import glob

import os

from general_utils import get_prefixes

class Models:

    def __init__(self,scaler):
        self.model=None
        self.scaler=scaler

    def InitLogistic(self,max_iter=10000,zero_weight=None):
        weights = {0: zero_weight}
        if zero_weight is None:
            weights='balanced'
        
        #self.model=OneVsRestClassifier(LogisticRegression(max_iter=max_iter,solver="lbfgs",class_weight=weights),n_jobs=-1)
        self.model = SGDClassifier(loss='log_loss', max_iter=max_iter, class_weight=weights)

    def InitSVM(self,zero_weight=None):
        weights = {0: zero_weight}
        if zero_weight is None:
            weights='balanced'
        #self.model=OneVsRestClassifier(SVC(kernel='rbf',C=1.0,gamma='scale',probability=True))
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, class_weight=weights)

    def train_model(self,X_train,y_train,scaled_X=True,thr=0.1):
        if not scaled_X:
            X_train=self.scaler.transform(X_train)

        if y_train.ndim==2:
            y_train=np.asarray([self.one_hot_class(x,thr) for x in y_train])

        #print(y_train)
        #y_train=np.asarray([self.one_hot_class(x,thr) for x in y_train])

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

    def evaluate_metric(self,y_test,y_pred,one_hot=False,thr=0.1,classes_str=['None']):
        
        if one_hot:
            y_test=np.asarray([self.one_hot_class(x,thr) for x in y_test])
            #y_pred=np.asarray([self.one_hot_class(x,thr) for x in y_pred])
        
        else:
            y_pred=np.asarray([round(x) for x in y_pred])

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        return acc, f1, report

    def predict(self,X,scaled_X=False):
        if not scaled_X:
            X=self.scaler.transform(X)
        
        return self.model.predict(X)

    def predict_proba(self,X,scaled_X=False):
        if not scaled_X:
            X=self.scaler.transform(X)
        return self.model.predict_proba(X)

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
    def process_one(directory,prefix,type_,save_dir,thr=0.1,zero_weight=None,zero_weight_h=None):
        scaler = joblib.load(directory + f'/{prefix}_scaler.pkl')
        m=Models(scaler)

        if type_=='SVM':
            m.InitSVM(zero_weight=zero_weight[int(prefix)])
        elif type_=="Logistic":
            m.InitLogistic(zero_weight=zero_weight[int(prefix)])
        else:
            raise NotImplementedError("Select a model (SVM or Logistic)")

        os.makedirs(save_dir,exist_ok=True)

        save_path=save_dir+'/'+str(prefix)

        with open(f'{directory}/{prefix}.json') as f:
            scheme=json.load(f)

        b=BalancedSet()
        TrainModels.trainmodel(b,m,f'{directory}/{prefix}',save_path,head=False,thr=thr)

        if scheme['predict_head']:
            
            if type_=='SVM':
                m.InitSVM(zero_weight=zero_weight_h[int(prefix)])
            elif type_=="Logistic":
                m.InitLogistic(zero_weight=zero_weight_h[int(prefix)])
            else:
                raise NotImplementedError("Select a model (SVM or Logistic)")

            TrainModels.trainmodel(b,m,f'{directory}/{prefix}_heads',save_path+'_heads',head=True)

        shutil.copyfile(f'{directory}/{prefix}.json',save_path+'.json')
        shutil.copyfile(f'{directory}/{prefix}_bow.json',save_path+'_bow.json')
        shutil.copyfile(f'{directory}/{prefix}_scaler.pkl',save_path+'_scaler.pkl')
        #
        

    @staticmethod
    def process_all(directory,type_,save_dir,thr=0.1,zero_weight=None,zero_weight_h=None):
        prefixes=get_prefixes(directory)
        for prefix in prefixes:
            TrainModels.process_one(directory,prefix,type_,save_dir,thr=thr,zero_weight=zero_weight,zero_weight_h=zero_weight_h)
        

if __name__=="__main__":

    TrainModels.process_all('bigger_balanced','SVM','bigger_models',thr=0.1)
