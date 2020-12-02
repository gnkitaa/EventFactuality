import pandas as pd
import re, sys, os
import numpy as np

def get_index(text_with_markup):
    source_idx = 'NA'
    event_idx = 'NA'
    for idx, token in enumerate(text_with_markup.split(" ")):
        if(token=='Author'):
            source_idx = idx
        elif('<' in token):
            source_idx = idx
        elif('[' in token):
            event_idx = idx
    return source_idx, event_idx

def get_source(text):
    s = re.search(r"<font color=\"Crimson\"><b>\S+</b></font>", text)
    if s:
        s = s.group(0)
        source = re.sub('<[^<]+?>','', s)
        if(source=='AUTHOR'):
            source = 'Author'
    return source

def get_event(text):
    s = re.search(r"<font color=\"MediumBlue\"><b>[\S]+</b></font>", text)
    if s:
        s = s.group(0)
        event = re.sub('<[^<]+?>', '', s)
    return event


def replace_html(text, keep_markup=False, keep_author=False):
    # replace source
    s = re.search(r"<font color=\"Crimson\"><b>\S+</b></font>", text)
    if s:
        s = s.group(0)
        source = re.sub('<[^<]+?>', '', s)
        if(keep_markup):
            text = re.sub(s, "<" + source + ">", text)
        else:
            text = re.sub(s, source, text)
    else:
        if(keep_author):
            text = "Author " + text
        
    # replace event
    s = re.search(r"<font color=\"MediumBlue\"><b>[\S]+</b></font>", text)
    if s:
        s = s.group(0)
        event = re.sub('<[^<]+?>', '', s)
        if(keep_markup):
            text = re.sub(s, "[" + event + "]", text)
        else:
            text = re.sub(s, event, text)
    else:
        return ""
    return text

def replace_html_dialouge_style(text, keep_markup=False):
    # replace source
    s = re.search(r"<font color=\"Crimson\"><b>\S+</b></font>", text)
    if s:
        s = s.group(0)
        source = re.sub('<[^<]+?>', '', s)
        if(keep_markup):
            text = re.sub(s, "<" + source + ">", text)
        else:
            text = re.sub(s, source, text)
    text = "Author : " + text
        
    # replace event
    s = re.search(r"<font color=\"MediumBlue\"><b>[\S]+</b></font>", text)
    if s:
        s = s.group(0)
        event = re.sub('<[^<]+?>', '', s)
        if(keep_markup):
            text = re.sub(s, "[" + event + "]", text)
        else:
            text = re.sub(s, event, text)
    else:
        return ""
    return text

def get_label(pos, neg, uu, na):
    return np.argmax(np.array([pos, neg, uu, na]))

def print_distribution(data, name):
    print("\n")
    print(name, " Data Distribution")
    class_dict = {0:'positive', 1:'negative', 2:'uncommitted', 3:'not_applicable'}
    for i in range(4):
        print("Class {}: {} , Percentage = {}%, Count = {}".format(i, class_dict[i], \
        np.round(100*len(data[data.label==i])/len(data), 2), len(data[data.label==i])))       
    prompts = set(data.clean_sentence)
    print('Number of unique prompts : ', len(prompts))


def process_data(infile, outdir, min_judge=3, thresh_pa_lb=0.0, thresh_pa_ub=1.0):
    
    raw_data          = pd.read_csv(infile)
    raw_filtered_data = raw_data[(raw_data.njudge>=min_judge)&\
                                 (raw_data.pa>=thresh_pa_lb)&\
                                 (raw_data.pa<=thresh_pa_ub)]
    raw_filtered_data = raw_filtered_data[~raw_filtered_data.source.str.contains("_")]
    
    source  = raw_filtered_data.apply(lambda row : get_source(row['source']), axis = 1).to_list()
    event   = raw_filtered_data.apply(lambda row : get_event(row['event']), axis = 1).to_list()
    label   = raw_filtered_data.apply(lambda row : get_label(row['pos'], row['neg'], \
                                        row['uu'], row['na']), axis = 1).to_list()
    
    sentence_clean = raw_filtered_data.apply(lambda row : \
                                                replace_html(row['sent']), \
                                                axis = 1).to_list()

    sentence_dialouge = raw_filtered_data.apply(lambda row : \
                                                replace_html_dialouge_style(row['sent'], \
                                                keep_markup=False), \
                                                axis = 1).to_list()
    
    sentence_dialouge_with_markup = raw_filtered_data.apply(lambda row : \
                                                replace_html_dialouge_style(row['sent'], \
                                                keep_markup=True), \
                                                axis = 1).to_list()
    
    
    
    filtered_data                                  = pd.DataFrame(columns=['original_prompt'])
    filtered_data['original_prompt']               = raw_filtered_data.sent
    filtered_data['pa']                            = raw_filtered_data.pa
    filtered_data['source']                        = source
    filtered_data['event']                         = event
    filtered_data['clean_sentence']                = sentence_clean
    filtered_data['sentence_dialouge']             = sentence_dialouge
    filtered_data['sentence_dialouge_with_markup'] = sentence_dialouge_with_markup
    filtered_data['label']                         = label
    filtered_data['positive']                      = raw_filtered_data.pos
    filtered_data['negative']                      = raw_filtered_data.neg
    filtered_data['uncommitted']                   = raw_filtered_data.uu
    filtered_data['not_applicable']                = raw_filtered_data.na
    filtered_data['isgold']                        = raw_filtered_data.isgold
    
    source_index, event_index   = zip(*filtered_data.apply(lambda row :\
                                get_index(row['sentence_dialouge_with_markup']),\
                                axis = 1).to_list()) 
    filtered_data['source_index'] = list(source_index)
    filtered_data['event_index'] = list(event_index)
    filtered_data = filtered_data.reset_index()
    filtered_data = filtered_data[filtered_data.source_index!='NA']
    assert len(filtered_data[filtered_data['clean_sentence'].isnull()])==0, "Null Sentences Found"
    filtered_data.to_csv(os.path.join(outdir, 'processed_data.csv'), sep='\t', index=False)
    
    gold_data = filtered_data[filtered_data.isgold==1]
    gold_prompts = set(gold_data.clean_sentence)

    non_gold_data = filtered_data[filtered_data.isgold==0]
    non_gold_prompts = set(non_gold_data.clean_sentence)
    print_distribution(non_gold_data, "Train")
    
    gold_prompts_only  = list(set(gold_prompts).difference(set(non_gold_prompts)))
    gold_data_only = gold_data[gold_data.clean_sentence.isin(gold_prompts_only)]
    print_distribution(gold_data_only, "Val")
    
    non_gold_data.to_csv(os.path.join(outdir, 'train_data.csv'), sep='\t', index=False)
    gold_data_only.to_csv(os.path.join(outdir, 'val_data.csv'), sep='\t', index=False)
    


if __name__ == "__main__":
    infile = sys.argv[1]
    outdir = sys.argv[2]
    process_data(infile, outdir, min_judge=3, thresh_pa_lb=0.0, thresh_pa_ub=1.0)