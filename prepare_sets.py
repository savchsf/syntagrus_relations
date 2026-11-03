from syntagrusbin import SynTagRusBinarizer

import json

import scalers

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss, TomekLinks

import torch

import numpy as np

import os

from glob import glob

import shutil

class BalancedSet:

    def __init__(self,scaler):
        self.scaler=scaler
        self.X_train=[]
        self.y_train=[]
        self.X_test=[]
        self.y_test=[]
        #self.prepare_sets_for_one() is called when not loaded from file

    def load_scheme(self,path): #or bow
        with open(path,encoding='utf-8') as f:
            return json.load(f)


    def load_from_scheme_for_eval(self,scheme,bow,data_path):
        scheme=self.load_scheme(scheme)
        bname=None
        if scheme['use_bert']:
            bname=scheme['bert_name']

        self.dataset=SynTagRusBinarizer(use_bert=scheme['use_bert'], bert_name=bname,
                                    batch_size=1,unk_token=scheme['unk_token'],use_head=scheme['use_head'],
                                    predict_head=scheme['predict_head'],use_pos=scheme['use_pos'],
                                    use_morph=scheme['use_morph'], use_sentenceemb=scheme['use_sentenceemb'],
                                    use_s_no=scheme['use_s_no'],use_distance=scheme['use_distance'],
                                    binary=scheme['binary'], sentence_batch=1
                                    )
        if not self.dataset.binary:
            self.dataset.data.all_rel=list(scheme['outputs'].values())

        bow=self.load_scheme(bow)
        self.dataset.all_words=bow["all_words"]
        self.dataset.all_pos=bow["all_pos"]
        self.dataset.all_rel=bow["all_rel"]
        self.dataset.all_features=bow["all_features"]

        self.dataset.load(data_path)

    def prepare_sets_rels(self,k=2,test_size=0.03,exp_path='./',prefix='0'): #split test,train

        #1: squeeze targets
        y=[self.dataset.squeeze_result(t) for t in self.dataset.targets]
        y=np.asarray(y)
        #2: train test
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.dataset.sources,y,test_size=test_size,stratify=y)
        #3: scale train
        self.X_train=self.scaler.fit_transform(self.X_train)
        #4: balance train
        nm3 = NearMiss(version=3, n_neighbors=k)
        nearmissed=False
        while (not nearmissed) and k>0:
            try:
                self.X_train, self.y_train = nm3.fit_resample(self.X_train, self.y_train)
                nearmissed=True
            except ValueError:
                k-=1

        tl = TomekLinks()
        try:
            self.X_train, self.y_train = tl.fit_resample(self.X_train, self.y_train)
        except Exception:
            pass

        #5: unsqueeze targets
        self.y_train=[self.dataset.depickle_result(x) for x in self.y_train]
        self.y_test=[self.dataset.depickle_result(x) for x in self.y_test]

        #return
        self.X_train=np.asarray(self.X_train)
        self.y_train=np.asarray(self.y_train)
        self.X_test=np.asarray(self.X_test)
        self.y_test=np.asarray(self.y_test)

        self.save_train_test(exp_path+'/'+prefix)

    def prepare_sets_head(self,k=2,test_size=0.03,exp_path='./',prefix='0'):

        datas=self.dataset.sources #np.array 2xn
        datas=np.asarray(datas)

        X=datas[:,:-1]
        y=datas[:, -1]

        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=test_size,stratify=y)
        #3: scale train
        self.X_train=self.scaler.transform(self.X_train)
        #4: balance train
        nm3 = NearMiss(version=3, n_neighbors=k)
        nearmissed=False
        while (not nearmissed) and k>0:
            try:
                self.X_train, self.y_train = nm3.fit_resample(self.X_train, self.y_train)
                nearmissed=True
            except ValueError:
                k-=1

        tl = TomekLinks()
        try:
            self.X_train, self.y_train = tl.fit_resample(self.X_train, self.y_train)
        except Exception:
            pass
        #return
        self.X_train=np.asarray(self.X_train)
        self.y_train=np.asarray(self.y_train)
        self.X_test=np.asarray(self.X_test)
        self.y_test=np.asarray(self.y_test)

        self.save_train_test(exp_path+'/'+prefix+'_heads')

    def export_dataset(self,scheme,bow,data_path,k=2,test_size=0.03,exp_path='./',prefix='0'):

        self.load_from_scheme_for_eval(scheme,bow,data_path)
        self.prepare_sets_rels(k,test_size,exp_path,prefix)

        if self.dataset.predict_head:
            self.prepare_sets_head(k,test_size,exp_path,prefix)

        self.scaler.save(exp_path+'/'+prefix+'_scaler.pkl')


    def save_train_test(self,path):
        torch.save(self.X_train,path+'_X_train.pt')
        torch.save(self.y_train,path+'_y_train.pt')
        torch.save(self.X_test,path+'_X_test.pt')
        torch.save(self.y_test,path+'_y_test.pt')

    def load_train_test(self,path):
        self.X_train=torch.load(path+'_X_train.pt',mmap=True,weights_only=False)
        self.y_train=torch.load(path+'_y_train.pt',mmap=True,weights_only=False)
        self.X_test=torch.load(path+'_X_test.pt',mmap=True,weights_only=False)
        self.y_test=torch.load(path+'_y_test.pt',mmap=True,weights_only=False)


if __name__=='__main__':
    scaler=scalers.MinMaxScaler()
    b=BalancedSet(scaler)
    p='bigger/0'
    b.export_dataset(p+'.json',p+'_bow.json',p,k=2,test_size=0.3,exp_path='bigger',prefix='0')
    print(b.y_train)
