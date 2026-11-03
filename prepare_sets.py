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

    def __init__(self):
        self.X_train=np.array([])
        self.y_train=np.array([])
        self.X_test=np.array([])
        self.y_test=np.array([])
        #self.prepare_sets_for_one() is called when not loaded from file

    def load_scheme(self,path): #or bow
        with open(path,encoding='utf-8') as f:
            return json.load(f)


    def load_from_scheme_for_eval(self,scheme,bow,data_path):
        '''
        class SynTagRusBinarizer(use_bert=True, bert_name='sergeyzh/rubert-large-uncased-sts',
        batch_size=32, unk_token='_', use_head=True,
        predict_head=True, use_pos=True, use_morph=True,
        use_sentenceemb=True, use_s_no=True, use_distance=True,
        binary=False)
        '''
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

    def prepare_sets_for_one(self,scaler,k=2,test_size=0.03,exp_path='./',prefix='0'): #split test,train

        #1: squeeze targets
        squ_targets=[self.dataset.squeeze_result(t) for t in self.dataset.targets]
        squ_targets=np.asarray(squ_targets)
        #print(squ_targets)
        #2: train test
        X_train,X_test,y_train,y_test=train_test_split(self.dataset.sources,squ_targets,test_size=test_size)
        #print(y_train[0])
        #3: scale train
        #X_train=scaler.fit_transform(X_train)
        #X_test=scaler.transform(X_test)
        #4: balance train
        nm3 = NearMiss(version=3, n_neighbors=k)
        nearmissed=False
        while (not nearmissed) and k>0:
            try:
                X_train, y_train = nm3.fit_resample(X_train, y_train)
                nearmissed=True
            except ValueError:
                k-=1

        tl = TomekLinks()
        try:
            X_train, y_train = tl.fit_resample(X_train, y_train)
        except Exception:
            pass

        #5: unsqueeze targets
        y_train=[self.dataset.depickle_result(x) for x in y_train]
        y_test=[self.dataset.depickle_result(x) for x in y_test]

        self.transfer_head()

        #return
        self.X_train=np.asarray(X_train)
        self.y_train=np.asarray(y_train)
        self.X_test=np.asarray(X_test)
        self.y_test=np.asarray(y_test)



        self.save_train_test(exp_path+'/'+prefix)

    def transfer_head(self):
        if self.dataset.predict_head:
            for i in range(len(self.y_train)):
                self.y_train[i].append(self.X_train[i][-1])
                if not self.dataset.use_head:
                    self.X_train[i]=self.X_train[i][:-1]
            for i in range(len(self.y_test)):
                self.y_test[i].append(self.X_test[i][-1])
                if not self.dataset.use_head:
                    self.X_test[i]=self.X_test[i][:-1]


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
    b=BalancedSet()
    p='bigger/0'
    b.load_from_scheme_for_eval(p+'.json',p+'_bow.json',p)
    b.prepare_sets_for_one(scaler,exp_path='bigger',prefix='0')
    print(b.X_train)
