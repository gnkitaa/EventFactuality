from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
from transformers import BertTokenizer

class FactualityDataset(Dataset):

    def __init__(self, filename, maxlen):

        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter = '\t')
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

        self.class_dict = {'positive':0, 'negative':1, 'uncommitted':2, 'not_applicable':3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence_dialouge']
        label = self.df.loc[index, 'label']
        source = self.df.loc[index, 'source']
        event = self.df.loc[index, 'event']
        source_idx = self.df.loc[index, 'source_index']
        event_idx = self.df.loc[index, 'event_index']

        prob_labels = list(self.df.loc[index, ['positive', 'negative', 'uncommitted', 'not_applicable']])

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        
        #Inserting the CLS and SEP token in the beginning and end of the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
        
        #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        #Converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(tokens_ids) 

        #Obtaining the attention mask i.e a tensor containing 1s for no 
        #padded tokens and 0s for padded ones 
        
        attn_mask = (tokens_ids_tensor != 0).long()
        
        source_idx = torch.tensor(source_idx)
        event_idx = torch.tensor(event_idx)

        return sentence, "@".join(tokens), source_idx, event_idx,\
               tokens_ids_tensor, attn_mask, label, torch.tensor(prob_labels)