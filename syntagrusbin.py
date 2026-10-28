from binarize import *
from conllu import *
import torch

import math
import random

from balance import Balance, Distance
import copy

#from collections import Counter

class SynTagRusBinarizer(Binarizer):
    def __init__(self,use_bert=True,bert_name='sergeyzh/rubert-large-uncased-sts',normalize=True, vec_result=True, one_hot=True,batch_size=32,unk_token='_'):
        super().__init__(use_bert,bert_name)
        self.sources=[]
        self.targets=[]
        self.data=ConlluDataset()
        self.use_bert=use_bert
        self.normalize=normalize
        self.one_hot=one_hot

        self.word_no=-1
        self.sen_no=-1

        self.words=[]
        self.sentences=[]
        self.s_idx=-1

        self.unk_token=unk_token

        self.batch_size=batch_size

        self.head_idx=None
        self.vec_result=vec_result

    def load_data(self,fp):
        self.data.compute(fp)

    def _find_w2(self,word):
        #word=d.tokens[i]
        sentence_n=word['sentence']
        w2_id=word['head'] #head=0 - no 2nd word
        if w2_id==0:
            return None
        #find:
        #'sentence'==sentence_n
        #'no_in_sentence'==w2_id

        for w in self.data.tokens:
            if w['sentence']==sentence_n and w['no_in_sentence']==w2_id:
                return w

        return None

    def _encode_word(self,word:str,normalize=True,bert=True):
        if bert:
            self.words.append(word)
            self.word_no+=1
            return [self.word_no]
        
        return [self._bag_of_words_item(word,self.data.all_words,normalize)]

    def _encode_discr(self,elem,all_elems,hot=True,normalize=True):
        if hot:
            return self._one_hot(elem,all_elems)
        return [self._bag_of_words_item(elem,all_elems,normalize=normalize)]

    def _pair_word(self,w1,w2):
        w1_emb=self._encode_word(w1['word'],normalize=self.normalize,bert=self.use_bert)
        w1_emb=torch.Tensor(w1_emb)
        w2_emb=torch.zeros(w1_emb.shape)
        if w2:
            w2_emb=self._encode_word(w2['word'],normalize=self.normalize,bert=self.use_bert)
            w2_emb=torch.Tensor(w2_emb)
        else:
            w2_emb=None
        return [w1_emb,w2_emb]

    def _pair_pos(self,w1,w2):
        w1_pos=self._encode_discr(w1['pos'],self.data.all_pos,hot=self.one_hot,normalize=self.normalize)
        w2_pos=[0]*len(w1_pos)
        if w2:
            w2_pos=self._encode_discr(w2['pos'],self.data.all_pos,hot=self.one_hot,normalize=self.normalize)
        return [w1_pos,w2_pos]

    def _pair_morph(self,w1,w2):
        w1_morph=self.vectorize_features(w1['features'],self.data.all_features,normalize=self.normalize,hot=self.one_hot)
        w2_morph=[0]*len(w1_morph)
        if w2:
            w2_morph=self.vectorize_features(w2['features'],self.data.all_features,normalize=self.normalize,hot=self.one_hot)
        return [w1_morph,w2_morph]

    def _get_len_sentence(self,sentence_id):
        words=[]
        for token in self.data.tokens:
            if token['sentence']==sentence_id:
                words.append(token['no_in_sentence'])

        return max(words)

    def _get_sentence_n(self,w1,w2):
        w1_no=w1['no_in_sentence']
        w2_no=0
        if w2:
            w2_no=w2['no_in_sentence']

        if self.normalize:
            lsent=self._get_len_sentence(w1['sentence'])
            w1_no=w1_no/lsent
            w2_no=w2_no/lsent


        return w1_no, w2_no

    def _encode_sentence(self,sentence):
        self.sentences.append(sentence)
        self.sen_no+=1
        return self.sen_no

    def _decode_bert(self,lst,batch_size=32):
        return self.bert_batch(lst,batch_size=batch_size)


    def generate_source(self,w1,
            use_pos=True, use_morph=True, use_sentenceemb=True,use_s_no=True, use_distance=True, use_head=True
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
        #chunk=d.tokens[i]
        result=[]
        w2=self._find_w2(w1)

        w1_no,w2_no=self._get_sentence_n(w1,w2)
        result.extend(self._pair_word(w1,w2))

        if use_pos:
            result.extend(self._pair_pos(w1,w2))
        if use_morph:
            result.extend(self._pair_morph(w1,w2))
        if use_sentenceemb:
            if not self.use_bert:
                raise Exception("Enable bert first")

            
            sentence=self.data.sentences[w1['sentence']]
            result.append(self._encode_sentence(sentence))
            self.s_idx=len(result)-1
        
        if use_head:
            result.append([1])
            self.head_idx=len(result)-1

        if use_s_no:
            result.extend([[w1_no],[w2_no]])
        if use_distance:
            result.append([abs(w1_no-w2_no)])

        #result=self._glue_all(result)

        return result



    def generate_target(self,chunk,binary=False):
        '''
        target: 0/1 if binary, [0,1] if not. if vec_result, binary is not used
        '''
        #chunk: d.tokens[i]
        rel=chunk['relation']

        if self.vec_result:
            return self._one_hot(rel,self.data.all_rel)

        rel=self._bag_of_words_item(rel,self.data.all_rel,normalize=self.normalize)
        if binary:
            if rel!=0:
                rel=1
        return rel

    def gen_dataset(self,
            use_pos=True, use_morph=True, use_sentenceemb=True, use_s_no=True, use_distance=True,
            binary=False, use_head=False
        ):
        for t in self.data.tokens:
            if t['head']!=0:
                self.sources.append(self.generate_source(t,use_pos=use_pos, use_morph=use_morph, use_sentenceemb=use_sentenceemb,
             use_s_no=use_s_no, use_distance=use_distance,use_head=use_head))
                self.targets.append(self.generate_target(t,binary=binary))

        #targets:
        #[[[0],0,0],[[0],0,0,0]] not glued yet
        curr_w=0
        curr_s=0

        if self.use_bert:
            word_bert=self._decode_bert(self.words,batch_size=self.batch_size)
            #print(word_bert)
            if use_sentenceemb:
                sen_bert=self._decode_bert(self.sentences,batch_size=self.batch_size)

            for i in range(len(self.sources)):
                #self.targets[i][0] w1
                #self.targets[i][1] w2
                #self.targets[i][self.s_idx] sen
                self.sources[i][0]=word_bert[curr_w]
                curr_w+=1
                self.sources[i][1]=word_bert[curr_w]
                curr_w+=1

                if use_sentenceemb:
                    self.sources[i][self.s_idx]=sen_bert[curr_s]
                    curr_s+=1

        


    def save(self,path): 
        export_data={
            "sources":self.sources,
            "targets":self.targets
        }
        torch.save(export_data,path)

    def load(self,path):
        loaded_data=torch.load(path)
        self.sources=loaded_data['sources']
        self.targets=loaded_data['targets']

    def generate_negative_data(self,ratio):
        number=len(self.data.tokens)
        gen_n=math.ceil(number*ratio)
        result=[]

        for sample in range(gen_n):
            random_s=random.randint(0,number-1)
            word=self.data.tokens[random_s]
            sentence_n=word['sentence']
            curr_rel=word['head']
            all_tokens_no_rel=[]

            for t in self.data.tokens:
                if t['sentence']==sentence_n and t['head']!=curr_rel:
                    all_tokens_no_rel.append(t)
            generate_n=random.randint(0,len(all_tokens_no_rel)-1)
            new_token=copy.deepcopy(word)
            new_token['head']=all_tokens_no_rel[generate_n]['no_in_sentence']
            new_token['relation']=self.unk_token
            result.append(new_token)
            #print(new_token)
        #print(result)

        self.data.tokens.extend(result)

    def _shuffle_head_one(self,sample):
        #idx: 0-not head, 1-head,self.head_idx
        #only call if use_head
        if self.head_idx:
            sample[0],sample[1]=sample[1],sample[0]
            sample[self.head_idx]=(sample[self.head_idx]+1)%2
        return sample
    
    def shuffle_head(self,sources,chance=0):
        for i in range(len(sources)):
            r=random.random()
            if r<chance:
                sources[i]=self._shuffle_head_one(sources[i])
        return sources

    def shuffle_data(self):
        indices=list(range(len(self.targets)))
        random.shuffle(indices)
        new_s, new_t= [],[]
        for i in indices:
            new_s.append(self.sources[i])
            new_t.append(self.targets[i])
        self.sources=new_s
        self.targets=new_t

    def _gen_neg_head_one(self,sample,tsample):
        if self.head_idx:
            sample[self.head_idx]=(sample[self.head_idx]+1)%2
            if isinstance(tsample,list):
                tsample[self.head_idx]=[0]*len(tsample)
            else:
                tsample=0
        return sample,tsample

    def generate_negative_head(self,ratio):
        for i in range(len(self.sources)):
            r=random.random()
            if r<ratio:
                self.sources[i],self.targets[i]=self._gen_neg_head_one(self.sources[i],self.targets[i])

    @staticmethod
    def _flat4dist(l):
        l=SynTagRusBinarizer._glue_all(l) #tensor
        return l.tolist()

    @staticmethod
    def _scaler_norm(l):
        mu=(sum(l))/(len(l))
        sigma=0
        for i in l:
            sigma+=(i-mu)**2
        sigma=(sigma/len(l))**0.5
        return [(x-mu)/sigma for x in l]


    def get_major_minor_classes(self):
        #ХЗ
        major_classes=[]
        minor_classes=[]
        counts={}
        was_list=False

        for sample in self.targets:
            key=sample
            if (isinstance(key,list)):
                key=tuple(key)
                was_list=True
            if key not in counts:
                counts[key]=0
            counts[key]+=1

        equal_percent=1/len(counts)
        n=len(self.targets)
        
        for class_, count in counts.items():
            var=count/n
            if was_list:
                class_=list(class_)
            if var>=equal_percent:
                major_classes.append(class_)
            else:
                minor_classes.append(class_)
        
        major_classes.sort()
        minor_classes.sort()

        return major_classes, minor_classes

    def balance_dataset(self,dist=Distance.Euclidean,nm_k=3,tl_delta=0.2):

        
        major_classes,minor_classes=self.get_major_minor_classes()
        #print(len(major_classes))
        #print(len(minor_classes))

        

        balance_instance=Balance(self.sources,self.targets,self._flat4dist,self._scaler_norm)

        for minor in minor_classes:
            for major in major_classes:
                balance_instance.NearMiss3(major,minor,dist,nm_k)
                balance_instance.TomekLinks(major,minor,Distance.Cosine,tl_delta)

        self.sources=balance_instance.sources
        self.targets=balance_instance.targets


    def generate_dataset(self,use_head=False,do_shuffle=True,neg_ratio=0.3,predict_head=True,
                        use_pos=True, use_morph=True, use_sentenceemb=True, use_s_no=True, use_distance=True,
                        binary=False):
        #final dataset: aug (?), neg, shuffle, balance, move head to targets, gluing the tensors
        #gen_dataset
        #....
        #use_head - использовать head в SOURCES
        '''
        use_pos=True, use_morph=True, use_sentenceemb=True, use_s_no=True, use_distance=True,
            binary=False, use_head=False
        '''

        if use_head and not(predict_head):
            neg_ratio/=2

        self.generate_negative_data(neg_ratio)

        self.gen_dataset(use_pos=use_pos,use_morph=use_morph,use_sentenceemb=use_sentenceemb,use_s_no=use_s_no,
                        use_distance=use_distance, binary=binary,use_head=(use_head or predict_head))
        
        #print([len(self._flat4dist(x)) for x in self.sources])

        if use_head and not(predict_head):
            self.generate_negative_head(neg_ratio)
            

        if do_shuffle:
            self.shuffle_data()

        #balance here....

        #print([len(self._flat4dist(x)) for x in self.sources])


        #print(len(self.targets))
        self.balance_dataset()
        #print(len(self.targets))

        if predict_head:
            for i in range(len(self.sources)): #перекидывание головы
                #print(self.sources[i][self.head_idx])
                head_param=self.sources[i][self.head_idx][0]
                self.sources[i]=list(self.sources[i])
                self.sources[i].pop(self.head_idx)
                if self.vec_result:
                    self.targets[i].append(head_param)
                else:
                    self.targets[i]=[self.targets[i],head_param]

        for i in range(len(self.sources)):
            self.sources[i]=self._glue_all(self.sources[i]).numpy()
            self.targets[i]=torch.Tensor(self.targets[i]).numpy()
        
if __name__=='__main__':
    b=SynTagRusBinarizer(use_bert=False,batch_size=64,vec_result=True)
    b.load_data('dataset/slice.conllu')

    b.generate_dataset(use_sentenceemb=False)
    #print(b.sources,b.targets)
    print(b.sources[-1])
    print(b.targets[-1])

    b.save('here.pt')
