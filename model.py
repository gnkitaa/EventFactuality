import torch
import torch.nn as nn
from transformers import BertModel

class Classifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(Classifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', \
                                                    output_hidden_states = True)
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(1536, 4)
        
    def get_word_embeddings(self, tokenized_text, tokenized_embedding):
        '''
        average the sub-token embeddings to get word embeddings
        '''
        word_embeddings = []
        idx = 0
        while(idx<len(tokenized_text)):
            cur = tokenized_embedding[idx]
            h_idx = idx+1
            count = 1
            while((h_idx<len(tokenized_text)) and ('#' in tokenized_text[h_idx])):
                cur = cur + tokenized_embedding[h_idx]
                count+=1
                h_idx+=1
            cur = cur/count
            word_embeddings.append(cur)
            idx = idx+count
        word_embeddings = torch.stack(word_embeddings, dim=0)
        return word_embeddings

    def forward(self, seq, tokens, s_idx, e_idx, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks 
                          to be used to avoid contibution of PAD tokens
            -s_idx : tensor of shape [B] containing index of source in seq
            -e_idx : tensor of shape [B] containing index of event in seq
            -tokens: string containing textual tokens obtained from bert tokenizer.
                     format of string t1@t2@t3....@tn
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _, hidden_states = self.bert_layer(seq, attention_mask = attn_masks)
 
        #Obtaining the representation of [CLS] head
        #cls_rep = cont_reps[:, 0]

        #Obtaining token embeddings from last layer
        token_embeddings = torch.stack(hidden_states, dim=0)[-1]
        
        token_text = []
        for t in tokens:
            token_text.append(t.split('@'))
            
        word_embeddings = []
        for te, emb in zip(token_text , token_embeddings):
            word_embeddings.append(self.get_word_embeddings(te, emb))
            
        
        se_embeddings = []
        for b in range(len(word_embeddings)):
            word_emb = word_embeddings[b]
            source_emb = word_emb[s_idx[b]]
            event_emb = word_emb[e_idx[b]]
            emb = torch.cat((source_emb, event_emb))
            se_embeddings.append(emb)
        se_embeddings = torch.stack(se_embeddings, dim=0)
        
        if(se_embeddings.shape[1]!=1536):
            print(torch.stack(hidden_states, dim=0).shape, token_embeddings.shape)

        logits = self.cls_layer(se_embeddings)

        return logits