from natasha import (Doc,
                     Segmenter,NewsEmbedding,
                     NewsMorphTagger)


class FeatExtractor:
    self.segmenter=Segmenter
    self.emb=NewsEmbedding()
    self.morph_tagger=NewsMorphTagger(emb)

    def __init__(self,unk_token='_'):
        self.unk_token=unk_token

    def _segment(self,sen):
        sen.segment(Segmenter)
        return sen

    def info(self,word,sen_doc_segmented): #(pos, features)
        sen_doc_segmented.tag_morph(morph_tagger)
        for token in sen_doc_segmented.tokens:
            if token.text.lower()==word.lower():
                return (token.pos,token.feats)
        return (self.unk_token,{self.unk_token:self.unk_token})

    def present(self,word, sentence):
        return word.lower() in sentence.lower()

    def sanity_check(self,aware_word_list,sentence):
        bl=[self.present(x,sentence) for x in aware_word_list]
        return not (False in bl)

    def tokenize(self,sentence,aware_word_list=[]):
        aware_word_list=[x.lower() for x in aware_word_list]
        if not self.sanity_check(aware_word_list,sentence):
            raise ValueError("Sentence does not contain such words")
        doc=Doc(sentence)
        doc.segment(self.segmenter)
        return doc

    def find_num(self, word, sentence_doc):
        c=1
        for i in sentence_doc.tokens:
            if i.text.lower()==word.lower():
                return c
            c+=1
        return 0

    def sentence_len(self,sentence_doc):
        return len(sentence_doc.tokens)

