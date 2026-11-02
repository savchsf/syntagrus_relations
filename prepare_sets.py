from syntagrusbin import SynTagRusBinarizer

import json

import scalers

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss, TomekLinks

import torch

import numpy as np

class BalancedSet:

    def __init__(self,scheme='.',path='.',scaler=None,test_size=0.2,k=3):
        self.X_train=np.array([])
        self.y_train=np.array([])
        self.X_test=np.array([])
        self.y_test=np.array([])

        self.scaler=scaler
        self.test_size=test_size


        self.dataset=None

        self.k=k

        #self.prepare_sets_for_one() is called when not loaded from file

    def load_scheme(self,path):
        with open(path,encoding='utf-8') as f:
            return json.load(f)

    def load_from_scheme_for_eval(self,scheme,data_path):
        '''
        class SynTagRusBinarizer(use_bert=True, bert_name='sergeyzh/rubert-large-uncased-sts',
        batch_size=32, unk_token='_', use_head=True,
        predict_head=True, use_pos=True, use_morph=True,
        use_sentenceemb=True, use_s_no=True, use_distance=True,
        binary=False)
        '''
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

        self.dataset.all_words=scheme["all_words"]
        self.dataset.all_pos=scheme["all_pos"]
        self.dataset.all_rel=scheme["all_rel"]
        self.dataset.all_features=scheme["all_features"]

        self.dataset.load(data_path)

    def prepare_sets_for_one(self): #split test,train
        #1: squeeze targets
        squ_targets=[self.dataset.squeeze_result(t) for t in self.dataset.targets]
        squ_targets=np.asarray(squ_targets)
        #print(squ_targets)
        #2: train test
        X_train,X_test,y_train,y_test=train_test_split(self.dataset.sources,squ_targets,test_size=self.test_size)
        #print(y_train[0])
        #3: scale train
        for i in range(len(X_train)):
            self.scaler.fit(X_train[i])
            X_train[i]=self.scaler.transform(X_train[i])
        #4: balance train
        nm3 = NearMiss(version=3, n_neighbors=self.k)
        nearmissed=False
        while (not nearmissed) and self.k>0:
            try:
                X_train, y_train = nm3.fit_resample(X_train, y_train)
                nearmissed=True
            except ValueError:
                self.k-=1

        tl = TomekLinks()
        X_train, y_train = tl.fit_resample(X_train, y_train)

        #5: unsqueeze targets
        y_train=[self.dataset.depickle_result(x) for x in y_train]
        y_train=np.asarray(y_train)
        y_test=[self.dataset.depickle_result(x) for x in y_test]
        y_test=np.asarray(y_test)
        #return
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test

    def save_train_test(self,path):
        export_data={
                "X_train":self.X_train,
                "y_train":self.y_train,
                "X_test":self.X_test,
                "y_test":self.y_test
            }
        torch.save(export_data,path)

    def load_train_test(self,path):
        loaded_data=torch.load(path,weights_only=False)
        self.X_train=loaded_data["X_train"]
        self.y_train=loaded_data["y_train"]
        self.X_test=loaded_data["X_test"]
        self.y_test=loaded_data["y_test"]


if __name__=='__main__':
    scaler=scalers.MinMaxScaler()
    b=BalancedSet('slice/0.json','slice/0.pt',scaler)
    self.scheme=self.load_scheme(scheme)
    self.load_from_scheme_for_eval(self.scheme,path)
    b.prepare_sets_for_one()
    b.save_train_test('slice/train_test_0.pt')
    print(b.X_train)
