from syntagrusbin import SynTagRusBinarizer

import json

import scalers

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import (NearMiss, TomekLinks, RandomUnderSampler, EditedNearestNeighbours,
                                     InstanceHardnessThreshold
                                     )
from imblearn.over_sampling import (ADASYN, SMOTE, RandomOverSampler)

import torch

import numpy as np

import os

from glob import glob

import shutil

from collections import Counter

from general_utils import get_prefixes
import copy

all_balance_mathods=["NearMiss1","NearMiss2","NearMiss3",
                     'TomekLinks',"RandomUnderSampler","EditedNearestNeighbours",
                     "InstanceHardnessThreshold",
                     "ADASYN","SMOTE","RandomOverSampler"]

class BalancedSet:
    def __init__(self):
        
        self.X_train=[]
        self.y_train=[]
        self.X_test=[]
        self.y_test=[]

    def set_scaler(self,scaler):
        self.scaler=scaler

    def load_scheme(self,path): #or bow
        with open(path,encoding='utf-8') as f:
            return json.load(f)


    def load_from_scheme_for_eval(self,scheme,data_path,data_path_val):

        print(data_path,data_path_val)

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

        self.dataset_val=copy.deepcopy(self.dataset)

        self.dataset.all_rel=scheme['outputs']

        self.dataset.load(data_path)

        self.dataset_val.load(data_path_val)

    def _NearM(self,k=2,ver=3):
        nm3 = NearMiss(version=ver, n_neighbors=k)
        nearmissed=False
        while (not nearmissed) and k>0:
            try:
                self.X_train, self.y_train = nm3.fit_resample(self.X_train, self.y_train)
                nearmissed=True
            except ValueError:
                k-=1

        del nm3
    def _methd(self,method):
        try:
            self.X_train, self.y_train = method.fit_resample(self.X_train, self.y_train)
        except Exception:
            pass

    def do_balance_one(self,k=2,method='RandomUndersampling'):
        
        match method:
            case 'ADASYN':
                asn=ADASYN(n_neighbors=k)
                self._methd(asn)
                del asn
            
            case 'SMOTE':
                sm=SMOTE(k_neighbors=k)
                self._methd(sm)
                del sm

            case 'RandomOverSampler':
                ros=RandomOverSampler()
                self._methd(ros)
                del ros

            case 'NearMiss1':
                self._NearM(k,ver=1)

            case 'NearMiss2':
                self._NearM(k,ver=2)

            case 'NearMiss3':
                self._NearM(k,ver=3)

            case 'TomekLinks':
                tl = TomekLinks()
                self._methd(tl)
                del tl
            
            case 'RandomUnderSampler':

                ru=RandomUnderSampler()
                self._methd(ru)
                del ru
            
            case 'EditedNearestNeighbours':
                enn=EditedNearestNeighbours(n_neighbors=k)
                self._methd(enn)
                del enn
            
            case 'InstanceHardnessThreshold':
                iht=InstanceHardnessThreshold(cv=k)
                self._methd(iht)
                del iht

    def do_balance(self,k=2,methods=all_balance_mathods):
        '''
        all_balance_mathods=["NearMiss1","NearMiss2","NearMiss3",
                     'TomekLinks',"RandomUnderSampler","EditedNearestNeighbours",
                     "InstanceHardnessThreshold",
                     "ADASYN","SMOTE","RandomOverSampler",
                     "BalanceCascade","BalancedRandomForestClassifier"]
        '''
        t=[self.do_balance_one(k=k,method=x) for x in methods]

        


    def finalize(self,path):
        self.X_train=np.asarray(self.X_train)
        self.y_train=np.asarray(self.y_train)
        self.X_test=np.asarray(self.X_test)
        self.y_test=np.asarray(self.y_test)

        self.save_train_test(path)

    def prepare_sets_rels(self,k=2,test_size=0.03,exp_path='./',prefix='0',balance=all_balance_mathods): #split test,train

        X=self.dataset.sources
        X=np.asarray(X, dtype=np.float32)

        #1: squeeze targets
        y=[self.dataset.squeeze_result(t) for t in self.dataset.targets]
        y=np.asarray(y)

        #2: train test
        #self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=test_size,stratify=y)s\

        self.X_train=X
        self.y_train=y

        X=self.dataset_val.sources
        X=np.asarray(X, dtype=np.float32)
        y=self.dataset_val.targets
        y=np.asarray(y)

        self.X_test=X
        self.y_test=y

        #3: scale train
        self.X_train=self.scaler.fit_transform(self.X_train)
        #4: balance train
        self.do_balance(k=k,methods=balance)
    
        #5: unsqueeze targets
        self.y_train=[self.dataset.depickle_result(x) for x in self.y_train]
        #self.y_test=[self.dataset.depickle_result(x) for x in self.y_test]

        #return
        self.finalize(exp_path+'/'+str(prefix))
        del self.X_train
        del self.y_train
        del self.X_test
        del self.y_test

    def prepare_sets_head(self,k=2,test_size=0.03,exp_path='./',prefix='0',balance=all_balance_mathods):

        datas=self.dataset.sources #np.array 2xn
        datas=np.asarray(datas)

        X=datas[:,:-1]
        y=datas[:, -1]

        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=test_size,stratify=y)
        #3: scale train
        self.X_train=self.scaler.transform(self.X_train)
        #4: balance train
        self.do_balance(k=k,methods=balance)
        #return

        self.finalize(exp_path+'/'+str(prefix)+'_heads')

        del self.X_train
        del self.y_train
        del self.X_test
        del self.y_test

    def export_dataset(self,scheme,data_path,k=2,test_size=0.03,exp_path='./',prefix='0',balance=all_balance_mathods, h_balance=all_balance_mathods,data_path_val=''):

        os.makedirs(exp_path,exist_ok=True)

        self.load_from_scheme_for_eval(scheme,data_path,data_path_val)
        self.prepare_sets_rels(k,test_size,exp_path,prefix,balance=balance)

        if self.dataset.predict_head:
            self.prepare_sets_head(k,test_size,exp_path,prefix,balance=h_balance)

        self.scaler.save(exp_path+'/'+str(prefix)+'_scaler.pkl')


    def save_train_test(self, path):
        np.save(path + '_X_train', self.X_train.astype(np.float32, copy=False))
        np.save(path + '_y_train', self.y_train)
        np.save(path + '_X_test',  self.X_test.astype(np.float32, copy=False))
        np.save(path + '_y_test',  self.y_test)

    def load_train_test(self, path):
        self.X_train = np.load(path + '_X_train.npy', mmap_mode='r')
        self.y_train = np.load(path + '_y_train.npy', mmap_mode='r')
        self.X_test  = np.load(path + '_X_test.npy',  mmap_mode='r')
        self.y_test  = np.load(path + '_y_test.npy',  mmap_mode='r')

class DatasetBalance:
    @staticmethod
    def process_subset(directory,prefix,scaler,test_size,k,exp_path,balance=all_balance_mathods,h_balance=[],data_path_val=''):
        b=BalancedSet()
        b.set_scaler(scaler)
        gen_path=directory+'/'+str(prefix)
        gen_path_val=data_path_val+'/'+str(prefix)
        b.export_dataset(gen_path+'.json',gen_path,test_size=test_size,k=k,exp_path=exp_path,prefix=prefix,balance=balance,h_balance=h_balance,data_path_val=gen_path_val)
        shutil.copyfile(gen_path+'.json',exp_path+'/'+str(prefix)+'.json')
        shutil.copyfile(gen_path+'_bow.json',exp_path+'/'+str(prefix)+'_bow.json')


    @staticmethod
    def process_all(directory,exp_path,
                    scaler,test_size=0.3,k=2,balance=None, h_balance=None,data_path_val=''
                    ):
        prefixes=get_prefixes(directory)

        if balance is None:
            balance=[]*len(prefixes)

        if h_balance is None:
            h_balance=[]*len(prefixes)

        c=0

        for prefix in prefixes:
            DatasetBalance.process_subset(directory,prefix,scaler,test_size,k,exp_path,balance=balance[c],h_balance=h_balance[c],data_path_val=data_path_val)
            c+=1

if __name__=='__main__':
    scaler=scalers.MinMaxScaler()
    DatasetBalance.process_all('bigger','bigger_balanced',scaler)

