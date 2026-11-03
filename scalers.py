import numpy as np
from abc import ABC, abstractmethod

class Scaler(ABC):
    @abstractmethod
    def fit(self,matrix):
        pass

    @abstractmethod
    def transform(self,matrix):
        pass

    def fit_transform(self,matrix):
        self.fit(matrix)
        return self.transform(matrix)


class MinMaxScaler(Scaler):
    def fit(self,matrix):
        self.min_v=np.min(matrix)
        self.length=np.max(matrix)-self.min_v
        if self.length==0:
            self.length=1

    def transform(self,matrix):
        return (matrix-self.min_v)/self.length

class StandardScaler(Scaler):
    def fit(self,matrix,thr=1e-5):

        self.mu=np.mean(matrix)
        self.sigma=np.std(matrix)
        if self.sigma==0:
            self.sigma=thr

    def transform(self,matrix):
        return (matrix-self.mu)/self.sigma


if __name__=="__main__":
    m=np.array([
                [-25.0, -15.0],
                [ -5.0,   5.0],
                [ 15.0,  25.0]
                ])
    sc=StandardScaler()
    print(sc.fit_transform(m))
