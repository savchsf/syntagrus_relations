


class ConlluDataset:
    def __init__(self,unk_token='_'):
        self.sentences=[]
        self.tokens=[]
        self.all_features={}
        self.all_words=[]
        self.all_pos=[]
        self.all_rel=[]
        self.unk_token=unk_token

    def _split_features(self,line:str):
        #feature: Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur
        features=line.split('|')
        data={}
        for i in features:
            i=i.split('=')
            if len(i)<2: #means no features at all, just _
                return dict()
            data[i[0]]=i[1]

        return data

    def _parse_word(self,line:str, sentence_no:int):
        #line: 8	специалистов	специалист	NOUN	_	Animacy=Anim|Case=Gen|Gender=Masc|Number=Plur	5	nmod	5:nmod:gen	_
        features=line.split('\t') #len=10
        if features[1]==self.unk_token:
            return {}

        return {
            'no_in_sentence': int(features[0]),
            'word': features[1],
            'lemma': features[2],
            'pos': features[3],
            'xpos': features[4], #not used
            'features': self._split_features(features[5]),
            'head': int(features[6]),
            'relation': features[7],
            'parent:relation': features[8],
            'extra': features[9],
            'sentence': sentence_no
        }

    def _parse_file(self,fp):

        curr_s=-1

        with open(fp, encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if line:
                    if line.startswith('# text = '):
                        line=line.replace('# text = ','')
                        self.sentences.append(line)
                        curr_s+=1
                    elif line[0]!='#':
                        parse_res=self._parse_word(line,curr_s)
                        if parse_res:
                            self.tokens.append(parse_res)

    def _find_all_features(self):
        for word in self.tokens:
            feature_list=word['features']
            #print(feature_list)
            if feature_list:
                for key in feature_list:
                    value=feature_list[key]

                    if key in self.all_features:
                        if value not in self.all_features[key]:
                            self.all_features[key].append(value)
                    else:
                        self.all_features[key]=[value]

    def _sort_dict_of_lists(self,d):
        d=dict(sorted(d.items()))
        for key in d:
            d[key].sort()
        return d

    def _sorted_key(self,d,k):
        return sorted(set(x[k] for x in d))

    def compute(self,fp):
        self._parse_file(fp)

        self._find_all_features()
        self._sort_dict_of_lists(self.all_features)

        self.all_words=self._sorted_key(self.tokens,'word')
        self.all_pos=self._sorted_key(self.tokens,'pos')
        self.all_rel=self._sorted_key(self.tokens,'relation')


'''d=ConlluDataset()
d.compute('dataset/ru_syntagrus-ud-test.conllu')


if __name__=='__main__':
    print(d.tokens[88]['features'])

    print(d.all_features)'''