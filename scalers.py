import numpy as np


class MinMaxScaler:
    def fit(self,vector):
        self.min_v=np.min(vector,axis=0)
        self.length=np.max(vector,axis=0)-self.min_v

    def transform(self,vector):
        if np.sum(self.length)==0:
            return np.zeros(len(vector))
        return (vector-self.min_v)/self.length

class StandardScaler:
    def fit(self,vector,thr=1e-5):
        self.mu=np.mean(data,axis=0 )
        self.sigma=np.std(data,axis=0 )
        self.sigma[self.sigma==0]=thr

    def transform(self,vector):
        return (vector-self.mu)/self.sigma