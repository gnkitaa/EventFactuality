import torch.nn as nn
import torch.nn.functional as F
import torch

def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=1)
    _, preds = probs.max(1)
    acc_val = torch.eq(preds, labels.squeeze()).float().mean()
    return acc_val, preds, probs

def load_ckp(checkpoint_fpath, model, args):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint)
    model.cuda(args.gpu)
    model.eval()
    return model

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def evaluate(net, criterion, dataloader, args):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    preds, pred_probs = [], []
    true_labels, prob_labels = [], []
    sentences = []
    losses = []

    with torch.no_grad():
        for sent, tokens, s_idx, e_idx, seq, attn_masks, labels, p_labels in dataloader:
            seq, attn_masks, labels, p_labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu), p_labels.cuda(args.gpu)
            s_idx, e_idx = s_idx.cuda(args.gpu), e_idx.cuda(args.gpu)
            
            logits = net(seq, tokens, s_idx, e_idx, attn_masks)
            
            if(args.soft_labels):
              input_labels = p_labels
            else:
              input_labels = labels.long()
            
            loss_per_sample = criterion(logits.squeeze(-1), input_labels, reduction='none')
            mean_loss += criterion(logits.squeeze(-1), input_labels).item()
            acc, pred, pred_p = get_accuracy_from_logits(logits, labels)
            mean_acc += acc.item()

            preds.append(pred.detach().cpu().numpy())
            pred_probs.append(pred_p.detach().cpu().numpy())

            true_labels.append(labels.detach().cpu().numpy())
            prob_labels.append(p_labels.detach().cpu().numpy())
            
            losses.append(loss_per_sample.detach().cpu().numpy())

            sentences.append(sent)
            count += 1

    return mean_acc / count, mean_loss / count, preds, pred_probs,\
            true_labels, prob_labels, sentences, losses