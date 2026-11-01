from binarize import *
from conllu import *
import torch

import math
import random
import copy
import json
import os
import numpy as np
from collections import Counter

class SynTagRusBinarizer(Binarizer):
    def __init__(self,use_bert=True,bert_name='sergeyzh/rubert-large-uncased-sts',
    batch_size=32,unk_token='_',
    use_head=True, predict_head=True, use_pos=True, use_morph=True,
    use_sentenceemb=True, use_s_no=True, use_distance=True, binary=False
    ):
        #новая стратегия: все признаки всегда кодируются как one_hot. Голова всегда последняя.abs
        #переброс головы случится после балансировки, которая проводится перед тренировкой в отдельном файле (и для сурсов, и для таргетов)
        super().__init__(use_bert,bert_name)
        self.sources=[]
        self.targets=[]
        self.data=ConlluDataset()
        self.use_bert=use_bert

        self.word_no=-1
        self.sen_no=-1

        self.words=[]
        self.sentences=[]
        self.s_idx=-1

        self.unk_token=unk_token

        self.batch_size=batch_size

        self.bert_size=None

        #dataset_generation
        self.use_head=use_head
        self.predict_head=predict_head
        self.use_pos=use_pos
        self.use_morph=use_morph
        self.use_sentenceemb=use_sentenceemb
        self.use_s_no=use_s_no
        self.use_distance=use_distance
        self.binary=binary

        self.sentence_emb_idx=None

        self.scheme=dict()

        #cache
        self.sentence_len_cache=[]
        self.sentence_tokens_cache={} #{no_sen:[token1 token2]}

        #bag of words caches
        self.bow_words={} #word:idx
        self.bow_pos={} #pos:idx
        self.bow_features={} #feature:idx kn all_features
        #delete encode_discr
        self.bow_rel={} #rel:idx

    def _bag_of_words_cache_init(self):
        '''
        self.bow_words={} #word:idx
        self.bow_pos={} #pos:idx
        self.bow_features={}
        '''
        self.bow_words = {x: i +1 for i, x in enumerate(self.data.all_words)}
        self.bow_pos = {x: i +1 for i, x in enumerate(self.data.all_pos)}
        self.bow_features = {x: i +1 for i, x in enumerate(self.data.all_features)}
        #print(self.bow_features)


    def _bag_of_words_rel_cache_init(self):
        self.bow_rel = {x: i +1 for i, x in enumerate(self.data.all_rel)}

    def _sentence_cache_init(self):
        for i in self.data.tokens:
            sen=i['sentence']
            if sen not in self.sentence_tokens_cache:
                self.sentence_tokens_cache[sen]=[]
            self.sentence_tokens_cache[sen].append(i)
        self.sentence_len_cache=[len(self.sentence_tokens_cache[x]) for x in self.sentence_tokens_cache]

    def _clear_cache(self):
        self.sentence_len_cache=[]
        self.sentence_tokens_cache={}
        self.bow_words={} #word:idx
        self.bow_pos={} #pos:idx
        self.bow_features={} #feature:idx kn all_features
        self.bow_rel={}

    def load_data(self,data):
        self.data=data
        self._sentence_cache_init()
        self._bag_of_words_cache_init()
        self._bag_of_words_rel_cache_init()

    def load_data_from_file(self,fp):
        self.data.compute(fp)
        self._sentence_cache_init()
        self._bag_of_words_cache_init()
        self._bag_of_words_rel_cache_init()

    def _find_w2(self,word):
        w2_id=word['head']
        sen=word['sentence']
        return [x for x in self.sentence_tokens_cache[sen] if x['no_in_sentence']==w2_id][0]

    def _encode_word(self,word:str):
        if self.use_bert:
            self.words.append(word)
            self.word_no+=1
            return self.word_no
        return self._bag_of_words_item(word,self.bow_words)

    '''def _encode_discr(self,elem,all_elems):
        return self._one_hot(elem,all_elems)'''

    def _pair_word(self,w1,w2):
        w1_emb=self._encode_word(w1['word'])
        w2_emb=self._encode_word(w2['word'])
        return [np.array([w1_emb]),np.array([w2_emb])]

    def _pair_pos(self,w1,w2):
        w1_pos=self._one_hot(w1['pos'],self.bow_pos)
        w2_pos=self._one_hot(w2['pos'],self.bow_pos)
        return [np.array(w1_pos),np.array(w2_pos)]

    def _pair_morph(self,w1,w2):
        w1_morph=self.vectorize_features(w1['features'],self.bow_features)
        w2_morph=self.vectorize_features(w2['features'],self.bow_features)
        return [np.array(w1_morph),np.array(w2_morph)]


    def _get_len_sentence(self,sentence_id):
        return self.sentence_len_cache[sentence_id]

    def _get_sentence_n(self,w1,w2):
        w1_no=w1['no_in_sentence']
        w2_no=w2['no_in_sentence']
        return w1_no, w2_no

    def _encode_sentence(self,sentence):
        self.sentences.append(sentence)
        self.sen_no+=1
        return [np.array([self.sen_no])]

    def _decode_bert(self,lst):
        return self.bert_batch(lst,batch_size=self.batch_size)

    def generate_source(self,w1
        ):
        '''
        source: (* - optional)

        word1
        word2
        word1pos *
        word2pos *
        morph1 *
        morph2 *
        sentenceemb *
        head* (0 or 1)
        w1_no *
        w2_no *
        distance *
        '''

        result=[]
        w2=self._find_w2(w1)
        w1_no,w2_no=self._get_sentence_n(w1,w2)
        #_pair_word
        result.extend(self._pair_word(w1,w2)) #[arr(132),arr(42)]

        if self.use_pos:
            result.extend(self._pair_pos(w1,w2)) #[arr(132),arr(42), arr, arr]

        if self.use_morph:
            result.extend(self._pair_morph(w1,w2)) #[arr(132),arr(42), arr, arr, arr, arr6]

        if self.use_sentenceemb:

            if not self.use_bert:
                raise Exception("Enable bert first")

            sentence=self.data.sentences[w1['sentence']]
            result.extend(self._encode_sentence(sentence)) #[arr(1),arr(2), arr3, arr4, arr5, arr6, arr7]

            self.sentence_emb_idx=len(result)-1 #!!!!!!!!

        if self.use_s_no: #no in sentence
            result.extend([np.array([w1_no]), np.array([w2_no])]) #[arr(1),arr(2), arr3, arr4, arr5, arr6, arr7, arr8, arr9]

        if self.use_distance:
            result.extend([np.array([abs(w1_no-w2_no)])]) #[arr9, arr10,arr11]

        if self.use_head or self.predict_head: 
            result.extend([np.array([1])])  #[arr9, arr10,arr11, arr12]

        return result #python list of np arrays -> [np.arr(1),np.arr(2),np.arr(7),....]


    def generate_target(self,chunk):
        rel=chunk['relation']
        if self.binary:
            rel=self._bag_of_words_item(rel,self.bow_rel)
            if rel!=0:
                rel=1
            return np.array([rel])
        return np.array(self._one_hot(rel,self.bow_rel))
        #singular np.array

    def return_bert(self):
        curr_w=-1
        curr_s=-1
        word_bert=self._decode_bert(self.words)
        if self.bert_size is None:
            self.bert_size=len(word_bert[0])
        #print(len(self.sources))
        if self.use_sentenceemb:
            sen_bert=self._decode_bert(self.sentences)

        for i in range(len(self.sources)):
            curr_w+=1
            #w1
            #print(curr_w)
            self.sources[i][0]=word_bert[curr_w]
            curr_w+=1
            #w2
            self.sources[i][1]=word_bert[curr_w]

            if self.use_sentenceemb:
                curr_s+=1
                self.sources[i][self.sentence_emb_idx]=sen_bert[curr_s]
    

    def _flatten_all(self):
        for i in range(len(self.sources)):
            #print(self.sources[i][0])
            #print(self.sources[i][2])
            self.sources[i]=np.concatenate(self.sources[i])


    #------ГЕНЕРАЦИЯ НЕГАТИВНЫХ ПРИМЕРОВ------

    def _GFOWIS(self,token):
        new_token=copy.deepcopy(token)

        sentence_n=new_token['sentence']
        new_token['rel']=self.unk_token

        #select
        others=[x for x in self.sentence_tokens_cache[sentence_n] if x!=token]
        other_one=random.choice(others)

        new_token['head']=other_one['no_in_sentence']

        return new_token

    def generate_from_other_words_in_sentence(self,ratio=0.5):
        #этот работает с данными из парсера
        n_new=math.ceil(len(self.data.tokens)*ratio)
        selected_tokens=random.sample(self.data.tokens,k=n_new)

        new_tokens=[self._GFOWIS(x) for x in selected_tokens]

        self.data.tokens.extend(new_tokens)

    def _one_negative_head(self,s,t):
        new_source=copy.deepcopy(s)
        new_source[-1]=(new_source[-1]+1)%2
        new_target=np.zeros(len(t))
        return new_source, new_target

    def generate_negative_heads(self,ratio=0.5):
        #работает со списками из np.array
        n_new=math.ceil(len(self.data.tokens)*ratio)
        selected_indices=random.sample(list(range(len(self.targets))),k=n_new)
        res_s=[]
        res_t=[]
        for i in selected_indices:
            cs,ct=self._one_negative_head(self.sources[i],self.targets[i])
            res_s.append(cs)
            res_t.append(ct)
        self.sources.extend(res_s)
        self.targets.extend(res_t)

    #------SHUFFLE------

    def shuffle_data(self):
        #со списками из np.array
        combinations=list(zip(self.sources, self.targets))
        random.shuffle(combinations)
        self.sources[:],self.targets[:]=zip(*combinations)

    def _invert_head(self,s,t):
        new_s=copy.deepcopy(s)
        new_t=copy.deepcopy(t)
        new_s[0],new_s[1]=new_s[1],new_s[0]
        new_s[-1]=(new_s[-1]+1)%2
        return new_s,new_t

    def shuffle_head(self,ratio=0.5):
        #со списками из np.array
        n_new=math.ceil(len(self.targets)*ratio)
        selected_indices=random.sample(list(range(len(self.targets))),k=n_new)
        res_s=[]
        res_t=[]
        for i in selected_indices:
            cs,ct=self._invert_head(self.sources[i],self.targets[i])
            res_s.append(cs)
            res_t.append(ct)
        self.sources.extend(res_s)
        self.targets.extend(res_t)

    def generate_dataset(self,ratio=0.3,do_shuffle=True): #head still not
        if self.use_bert:
            self._init_bert()

        if (self.use_head or self.predict_head) and self.use_sentenceemb:
            ratio=ratio/2

        if self.use_sentenceemb:
            self.generate_from_other_words_in_sentence(ratio)
        
        c=0
        non_root=len([x for x in self.data.tokens if x['head']!=0])

        for t in self.data.tokens:
            if t['head']!=0 and (t['relation'] in self.data.all_rel or t['relation']==self.unk_token):
                self.sources.append(self.generate_source(t))
                self.targets.append(self.generate_target(t))
                c+=1
            if c%1000==0:
                print(c,non_root)

        

        if self.use_bert:
            self.return_bert()
            self._unload_bert()

        if self.use_head or self.predict_head:
            self.generate_negative_heads(ratio)

        if do_shuffle:
            self.shuffle_data()
            if self.use_head or self.predict_head:
                self.shuffle_head()

        self._flatten_all()
        self.sources=np.asarray(self.sources)
        self.targets=np.asarray(self.targets)

    def save(self,path): 
        export_data={
            "sources":self.sources,
            "targets":self.targets
        }
        torch.save(export_data,path)

    def load(self,path):
        loaded_data=torch.load(path,weights_only=False)
        self.sources=loaded_data['sources']
        self.targets=loaded_data['targets']



    def _pack_scheme(self):
        scheme={
            "use_bert":self.use_bert,
            "use_pos":self.use_pos,
            "use_morph":self.use_morph,
            "use_sentenceemb":self.use_sentenceemb,
            "use_s_no":self.use_s_no,
            "use_distance":self.use_distance,
            "binary":self.binary,
            "use_head":self.use_head,
            "predict_head":self.predict_head,
            "unk_token":self.unk_token
        }

        if self.use_bert:
            scheme["bert_size"]=self.bert_size
            scheme["bert_name"]=self.bert_name

        scheme["outputs"]=dict()
        all_rel=[self.unk_token]+self.data.all_rel
        if not self.binary:
            for i in range(len(all_rel)-1):
                scheme["outputs"][i]=self.data.all_rel[i]

        self.scheme=scheme

    def export_scheme(self,path):
        with open(path,'w',encoding='utf-8') as f:
            json.dump(self.scheme,f,indent=4)
'''
def split_major_minor(n,classes_):
    majors=[]
    minors=[]
    counts=dict(Counter(classes_))

    equal_percent=1/len(counts)
        
    for class_, count in counts.items():
        var=count/n
        if var>=equal_percent:
            majors.append(class_)
        else:
            minors.append(class_)
        
    majors.sort()
    minors.sort()

    return majors, minors


def select_w_classes(data_p,classes_):
    #data tokens
    data=copy.deepcopy(data_p)
    data.tokens=[x for x in data.tokens if x['relation'] in classes_]
    data.reinit()
    data.tokens=copy.deepcopy(data_p.tokens)
    return data

def split_two(data,relations):
    rels=[x['relation'] for x in data.tokens]
    rels=[x for x in rels if x in relations]
    majors,minors=split_major_minor(len(rels),rels)
    return [select_w_classes(data,majors), select_w_classes(data,minors)]

def split_n(data,n):
    data_n=[]
    if n<=1:
        return [data]
    if n==2:
        return split_two(data,data.all_rel)
    else:
        data_2=split_two(data,data.all_rel) #major, minor
        data_n.extend(split_n(data_2[0],n//2+n%2))
        data_n.extend(split_n(data_2[1],n//2))
        return data_n

def make_multiple_datasets(fp,dataset_configuration,n=3):
    data=ConlluDataset()
    data.compute(fp)
    data_n=split_n(data,n)
    datasets=[]
    for data_pack in data_n:
        dataset=copy.deepcopy(dataset_configuration)
        dataset.load_data(data_pack)
        dataset.generate_dataset()
        datasets.append(dataset)
    return datasets

    '''
if __name__=='__main__':
    b=SynTagRusBinarizer(use_bert=True,batch_size=64,use_sentenceemb=True,predict_head=True)
    #b.load_data('dataset/slice.conllu')
    b.load_data_from_file('dataset/ru_syntagrus-ud-train-a.conllu')
    b.generate_dataset()
    '''datasets=make_multiple_datasets('dataset/ru_syntagrus-ud-train-a.conllu',b)
    c=1
    for d in datasets:
        d._pack_scheme()
        d.export_scheme(f'test/{c}.json')
        d.save(f'test/{c}.pt')
        print(len(d.targets))
        print(len(d.targets[0]))
        c+=1'''