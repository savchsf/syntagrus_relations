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

from glob import glob

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


    def load_from_scheme_for_eval(self,scheme,data_path):
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

        self.dataset.all_rel=scheme['outputs']

        self.dataset.load(data_path)

    def do_balance(self,k=2):
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

    def finalize(self,path):
        self.X_train=np.asarray(self.X_train)
        self.y_train=np.asarray(self.y_train)
        self.X_test=np.asarray(self.X_test)
        self.y_test=np.asarray(self.y_test)

        self.save_train_test(path)

    def prepare_sets_rels(self,k=2,test_size=0.03,exp_path='./',prefix='0'): #split test,train

        #1: squeeze targets
        y=[self.dataset.squeeze_result(t) for t in self.dataset.targets]
        y=np.asarray(y)
        #2: train test
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.dataset.sources,y,test_size=test_size,stratify=y)
        #3: scale train
        self.X_train=self.scaler.fit_transform(self.X_train)
        #4: balance train
        self.do_balance(k=k)
    
        #5: unsqueeze targets
        self.y_train=[self.dataset.depickle_result(x) for x in self.y_train]
        self.y_test=[self.dataset.depickle_result(x) for x in self.y_test]

        #return
        self.finalize(exp_path+'/'+prefix)

    def prepare_sets_head(self,k=2,test_size=0.03,exp_path='./',prefix='0'):

        datas=self.dataset.sources #np.array 2xn
        datas=np.asarray(datas)

        X=datas[:,:-1]
        y=datas[:, -1]

        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=test_size,stratify=y)
        #3: scale train
        self.X_train=self.scaler.transform(self.X_train)
        #4: balance train
        self.do_balance(k=k)
        #return

        self.finalize(exp_path+'/'+prefix+'_heads')

    def export_dataset(self,scheme,data_path,k=2,test_size=0.03,exp_path='./',prefix='0'):

        os.makedirs(exp_path,exist_ok=True)

        self.load_from_scheme_for_eval(scheme,data_path)
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

class DatasetBalance:
    @staticmethod
    def process_subset(directory,prefix,scaler,test_size,k,exp_path):
        b=BalancedSet()
        b.set_scaler(scaler)
        gen_path=directory+'/'+prefix
        b.export_dataset(gen_path+'.json',gen_path,test_size=test_size,k=k,exp_path=exp_path,prefix=prefix)
        shutil.copy(gen_path+'.json',exp_path+'/'+prefix+'.json')
        shutil.copy(gen_path+'_bow.json',exp_path+'/'+prefix+'_bow.json')

    @staticmethod
    def get_prefixes(directory):
        paths=glob(directory+'/*.json') #counting prefixes by json
        prefixes=[x.split('/')[-1].replace('.json','') for x in list(paths) if '_bow' not in x]
        return prefixes

    @staticmethod
    def process_all(directory,exp_path,
                    scaler,test_size=0.3,k=2
                    ):
        prefixes=DatasetBalance.get_prefixes(directory)

        for prefix in prefixes:
            DatasetBalance.process_subset(directory,prefix,scaler,test_size,k,exp_path)

if __name__=='__main__':
    scaler=scalers.MinMaxScaler()
    DatasetBalance.process_all('bigger','bigger_balanced',scaler)
    