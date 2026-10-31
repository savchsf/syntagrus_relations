import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class Binarizer:

    def __init__(self,use_bert=True,bert_name='sergeyzh/rubert-large-uncased-sts'):
        self.tokenizer = None
        self.model=None
        self.bert_name=bert_name

    def _init_bert(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_name,local_files_only=True)
            self.model = BertModel.from_pretrained(self.bert_name,local_files_only=True)
            
        except Exception:
            print("Downloading models...")
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
            self.model = BertModel.from_pretrained(self.bert_name)
        
        self.model.eval()

    def _bag_of_words_item(self,item,whole_list,normalize=True,cached=True):
        if item not in whole_list:
            return 0
        if cached:
            result=whole_list[item]
        else:
            result=whole_list.index(item)+1
        if normalize:
            result=result/len(whole_list)
        return result

    def bag_of_words_list(self,items,whole_list,normalize=True):
        vector=[]
        for i in items:
            vector.append(self._bag_of_words_item(i,whole_list,normalize))
        return vector

    def vectorize_features(self,features, all_features, normalize=True, hot=True, cached=True):
        if hot:
            vector=[0]*len(all_features)
            for feature in features:
                idx=0
                if feature in all_features:
                    if cached:
                        idx=all_features[feature]-1
                    else:
                        idx=all_features.index(feature)
                vector[idx]=1
            return vector
        return self.bag_of_words_list(features,all_features,normalize=cached,cached=cached)


    def _mean_embed(self,embeddings): #torch.Tensor
        #print(embeddings.shape)
        if embeddings.dim()==1:
            return embeddings
        return embeddings.mean(dim=min(1,embeddings.dim()-1)).squeeze(0) #в одномерный список длины 768 или какая длина у модели

    '''def _tokenize_bert(self,text):
        inputs = self.tokenizer(text, return_tensors='pt',padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        embeddings=self._mean_embed(embeddings)
        
        return embeddings'''

    def _pooler(self,embedding):
        if len(embedding.shape)>=3:
            return embedding[0,0,:] 
        return embedding[0,:]

    def _chunks(self,lst,n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    def _bert_batch_words(self,words:list):
        inputs = self.tokenizer(words, return_tensors='pt',padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        #[batch_size, sequence_length, hidden_size(768/1024/...)]

        outputs=[]

        for i in range(embeddings.size(0)):
            batch=embeddings[i]
            outputs.append(self._pooler(batch).numpy())

        return outputs

    def bert_batch(self,words:list,batch_size=32):
        batches=self._chunks(words,batch_size) 
        results=[]
        for b in batches:
            results.extend(self._bert_batch_words(b))
        return results  

    def _one_hot(self, value, all_values, cached=True):
        vector = [0]*len(all_values)
        if cached:
            if value in all_values:
                vector[all_values[value]-1] = 1
        else:
            if value in all_values:
                vector[all_values.index(value)] = 1
        return vector

    @staticmethod
    def _glue_all(results:list):
        return np.concatenate(results)

    

'''
b=Binarizer()

from conllu import *

target_word=d.tokens[88]
x= b._tokenize_bert(target_word['word'])
print(x)

print(
    b.bert_batch(['hello','how'])
)

'''