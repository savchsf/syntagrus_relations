import random

class Distance:
    @staticmethod
    def Euclidean(a:list,b:list):
        #print(len(a),len(b))
        r=0
        for i in range(len(a)):
            r+=(a[i]-b[i])**2
        return r**0.5

    @staticmethod
    def Manhattan(a:list,b:list):
        r=0
        for i in range(len(a)):
            r+=abs(a[i]-b[i])
        return r

    @staticmethod
    def Mink(a,b,p):
        r=0
        for i in range(len(a)):
            r+=(abs(a[i]-b[i]))**p 
        return r**(1/p)

    @staticmethod
    def Cosine(a,b):
        scal=0
        asq=0
        bsq=0
        for i in range(len(a)):
            scal+=a[i]*b[i]
            asq+=a[i]**2
            bsq+=b[i]**2
        return 1 - scal/(asq**0.5)/(bsq**0.5)


class Balance: #binary data
    def __init__(self,sources,targets,flat_func=(lambda x: x),scaler=(lambda x: x)):
        self.sources=sources
        self.targets=targets
        self.flat_func=flat_func
        self.scaler=scaler

    def sel_with_class(self,class_) -> [int]:
        res=[]
        for i in range(len(self.targets)):
            #print(self.targets[i]==class_)
            #print(class_)
            if self.targets[i]==class_ and self.sources[i] is not None:
                res.append(i)

        return res

    def mark_for_del(self,r:int):
        self.targets[r]=None
        self.sources[r]=None
    
    def _clear(self,l):
        return [x for x in l if x is not None]
    
    def clear(self):
        self.targets=self._clear(self.targets)
        self.sources=self._clear(self.sources)

    def d_src(self,distance,a,b):
        return distance(self.scaler(self.flat_func(self.sources[a])),self.scaler(self.flat_func(self.sources[b])))

    def RandomUndersampling(self,class1,class2,threshold=0):
        #class1 - major, class2 - minor

        class1_elems=self.sel_with_class(class1)
        class2_elems=self.sel_with_class(class2)

        size_minor=len(class2_elems)
        size_major=len(class1_elems)


        while (size_major-threshold)>size_minor:
            mn=self.sel_with_class(class1)
            r=random.choice(mn)
            self.mark_for_del(r)
            size_major-=1
        self.clear()

    def KBorder(self,class1,class2,distance,k):
        class1_elems=self.sel_with_class(class1)
        class2_elems=self.sel_with_class(class2)
        for idx in class2_elems:
            distances=[]
            for i in class1_elems:
                #(distance,n_sample)
                distances.append((self.d_src(distance,i,idx),i))
            distances.sort()
            k_neigh=distances[:k]
            for pair in k_neigh:
                r=pair[1]
                self.mark_for_del(r)
            class1_elems=self.sel_with_class(class1)
        self.clear()

    def NearMiss1(self,class1,class2,distance,k,k_to_leave=None):
        class1_elems=self.sel_with_class(class1)
        class2_elems=self.sel_with_class(class2)

        dists_and_idx1=[]

        if k_to_leave is None:
            k_to_leave=len(class2_elems)

        for major in class1_elems:
            distances=[]
            for i in class2_elems:
                #(distance,n_sample)
                distances.append(self.d_src(distance,major,i))

            distances.sort()
            distances=distances[:k]
            mean_dist=0
            if len(distances)!=0:
                mean_dist=sum(distances)/len(distances)
            dists_and_idx1.append((mean_dist,major))
        dists_and_idx1.sort()
        dists_and_idx1=dists_and_idx1[k_to_leave:]
        for r in dists_and_idx1:
            self.mark_for_del(r[1])
        self.clear()

    def NearMiss3(self,class1,class2,distance,k):
        class1_elems=self.sel_with_class(class1)
        class2_elems=self.sel_with_class(class2)

        for minor in class2_elems:
            distances=[]
            for i in class1_elems:
                distances.append((self.d_src(distance,i,minor),i))
            distances=distances[k:]
            for i in distances:
                r=i[1]
                self.mark_for_del(r)
                class1_elems=self.sel_with_class(class1)
        self.clear()

    def TomekLinks(self,class1,class2,distance,delta=0):
        class1_elems=self.sel_with_class(class1)
        class2_elems=self.sel_with_class(class2)
        to_delete=[]

        min_d_minors=[]

        for minor in class2_elems:
            min_d_minor=min(self.d_src(distance,x,minor) for x in class1_elems+class2_elems)
            min_d_minors.append(min_d_minor)

        for major in class1_elems:
            #all the distances
            min_d=min(self.d_src(distance,major,x) for x in class1_elems+class2_elems)

            for min_d_minor in min_d_minors:
                if abs(min_d_minor-min_d)<=delta:
                    to_delete.append(major)

        for r in to_delete:
            self.mark_for_del(r)
        self.clear()

#тесты
if __name__=='__main__':
    sources=[
        [0,4,5,7,2,1], [4,2,1,4,6,2], [3,5,1,35,72,1], [2,3,5,2,5,1], [1,2,3,45,5,6], [5,3,2,5,2,1], [8,3,2,5,1,9]
    ]

    targets=[
        0,               1,             0,               0,              1,              0,              0
    ]

    print(sources,targets)
    print(len(sources),len(targets))

    b=Balance(sources,targets)
    #RandomUndersampling, KBorder, NearMiss1, NearMiss3, TomekLinks
    d=Distance.Euclidean
    #b.RandomUndersampling(0,1)
    #b.KBorder(0,1,d,1)
    #b.NearMiss1(0,1,d,2)
    #b.NearMiss3(0,1,d,2)
    #b.TomekLinks(0,1,d,5)
    print(b.sources,b.targets)
    print(len(b.sources),len(b.targets))
