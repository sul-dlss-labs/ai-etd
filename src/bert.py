# Based in part on http://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/

from typing import List

from fastai.text import BaseTokenizer
from transformers import BertTokenizer

class FastAIBertTokenizer(BaseTokenizer):
    
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs)->None:
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __call__(self, *args, **kwargs)->FastAIBertTokenizer:
        return self
    
    def tokenizer(self, t:str) -> List[str]:
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]