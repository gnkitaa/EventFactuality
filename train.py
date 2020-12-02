import numpy as np
import os
import torch
from evaluation import *


def train(net, criterion, opti, train_loader, val_loader, args):

    if val_loader is None:
        best_state = None
    
    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    best_acc = 0
    best_loss = np.inf

    best_model_path = os.path.join(args.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(args.experiment_root, 'last_model.pth')

    
    
    for ep in range(args.max_eps):
        train_epoch_size = 0
        tr_mean_acc, tr_mean_loss = 0, 0

        for it, (sent, tokens, s_idx, e_idx, seq, attn_masks, labels, p_labels) in enumerate(train_loader):
            net.train()

            #Clear gradients
            opti.zero_grad()  
            
            #Converting these to cuda tensors
            seq, attn_masks, labels, p_labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu), p_labels.cuda(args.gpu)
            s_idx, e_idx = s_idx.cuda(args.gpu), e_idx.cuda(args.gpu)
            
            #Obtaining the logits from the model
            #print('computing logits')
            logits = net(seq, tokens, s_idx, e_idx, attn_masks)
            #print('logits computed')

            #Computing loss
            if(args.soft_labels):
              input_labels = p_labels
            else:
              input_labels = labels.long()
            tr_loss = criterion(logits.squeeze(-1), input_labels)

            #Backpropagating the gradients
            tr_loss.backward()

            #Optimization step
            opti.step()

            tr_acc, pred, probs = get_accuracy_from_logits(logits, labels)

            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())
            tr_mean_acc+= tr_acc.item()
            tr_mean_loss+= tr_loss.item()
            train_epoch_size+=1

        print("Epoch {} completed with {} iterations. Train Loss : {} Train Accuracy : {}".format(ep+1, it+1, \
                                                                    tr_mean_loss/train_epoch_size,\
                                                                    tr_mean_acc/train_epoch_size))
        if(val_loader):
            net.eval()
            v_mean_acc, v_mean_loss, preds, pred_probs, true_labels, prob_labels, sentences, l = evaluate(net, criterion, val_loader, args)
            
            val_loss.append(v_mean_loss)
            val_acc.append(v_mean_acc)

            print("Val Loss : {} Val Accuracy : {}".format(v_mean_loss, v_mean_acc))
            if v_mean_loss <= best_loss:
                model_to_save = net.module if hasattr(net, 'module') else net
                torch.save(model_to_save.state_dict(), best_model_path)
                best_acc = v_mean_acc
                best_loss = v_mean_loss
                best_state = net.state_dict()

    model_to_save = net.module if hasattr(net, 'module') else net
    torch.save(model_to_save.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(args.experiment_root, name + '.txt'), locals()[name])