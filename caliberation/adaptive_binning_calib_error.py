"""
"Adaptive Binning" algorithm by Nguyen and O'Connor 
(Algorithm 1)  https://arxiv.org/pdf/1508.05154.pdf

Author: Katie Keith
2020-09-13
"""
import numpy as np
from collections import defaultdict, Counter 

def sort_pairs(pred_label_pairs):
    """
    Input
        pred_label_pairs: list of tuples 
        
    Sorts by first element of the tuple, returns in assending order
    """
    return sorted(pred_label_pairs, key=lambda x: x[0])

def assign_bin_labels(sorted_pred_label_pairs, beta): 
    """
    b_k = floor((k-1)/beta)+ 1
    """
    
    #k = pair label 
    pair_label2bin_label = {}
    for pair_label, _ in enumerate(sorted_pred_label_pairs):
        k = pair_label + 1 #their k starts at 1 
        bin_label = np.floor((k-1)/beta)+ 1 
        pair_label2bin_label[pair_label] = int(bin_label)
        
    return pair_label2bin_label

def define_bins(pair_label2bin_label, beta): 
    bin2pairs = defaultdict(list)
    for pair_label, bin_label in pair_label2bin_label.items(): 
        bin2pairs[bin_label].append(pair_label)
        
    #from algorithm: If the last bin has size less than β, merge it with the second-to-last bin (if one exists). 
    bin2pairs = dict(bin2pairs)
    last_bin = len(bin2pairs)
    last_bin_pairs = bin2pairs[last_bin]
    len_last_bin = len(last_bin_pairs)
    if bin2pairs.get(last_bin-1) != None: 
        if len_last_bin < beta:
            bin2pairs[last_bin-1] = bin2pairs[last_bin-1] + last_bin_pairs
            del bin2pairs[last_bin] 
    return bin2pairs

def calc_per_bin_probs(bin2pairs, sorted_pred_label_pairs):
    q_i_hat_list = []
    p_i_hat_list = []
    
    for bbin, pairs in bin2pairs.items():
        p_list = []
        q_list = []
        for pair_indx in pairs: 
            q, p = sorted_pred_label_pairs[pair_indx]
            q_list.append(q)
            p_list.append(p)
        
        #calculate hat_{q_i}
        q_hat = np.mean(np.array(q_list))
        q_i_hat_list.append(q_hat)
           
        #calculate \hat_{p_i}
        p_hat = np.mean(np.array(p_list))
        p_i_hat_list.append(p_hat)
    return q_i_hat_list, p_i_hat_list

def calc_calib_error(q_i_hat_list, p_i_hat_list, bin2pairs, N): 
    assert len(q_i_hat_list) == len(p_i_hat_list) == len(bin2pairs)
    
    ssum = 0 
    for i, (q_i, p_i) in enumerate(zip(q_i_hat_list, p_i_hat_list)): 
        len_b_i = len(bin2pairs[i+1])
        squared_term = (q_i - p_i)**2
        ssum+= len_b_i * squared_term
    
    calib_error = np.sqrt(1/N*ssum)
    return calib_error


def graph_calibration_curve(q_i_hat_list, p_i_hat_list):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	plt.plot(p_i_hat_list, q_i_hat_list)
	plt.scatter(p_i_hat_list, q_i_hat_list)
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel('Prediction Strength')
	plt.ylabel('Empirical Frequency')
	plt.title("Calibration curve")

	#plot y=x line
	lims = [
	    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]
	ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
	ax.set_aspect('equal')

	plt.show() 
	return plt 


def create_pred_label_pairs(q_list, y_list): 
    """
    - Use the model to predict given the docids
    
    Returns: 
     [(q_i, y_i), ...] 
         - where q_i = probability out of logreg
         - y_i = "true" label averaged over multiple annotators 
    """
    pred_label_pairs = [(q, y) for q, y in zip(q_list, y_list)]
    return pred_label_pairs

def mean_calib_error_over_bins(q_list, y_list, calib_bin_range):
	pred_label_pairs = create_pred_label_pairs(q_list, y_list)

	all_calib_error = []
	print("num_bins", "beta", "calib_error")
	print("==="*10)
	for num_bins in calib_bin_range: #TODO, automate this list 
	    beta = int(np.round(len(y_list)/num_bins))
	    calib_error = adaptive_binning_algo(list(pred_label_pairs), beta, output_q_y_est=False)
	    print(num_bins, beta, calib_error)
	    all_calib_error.append(calib_error)
	all_calib_error = np.array(all_calib_error)
	print()
	mn = np.mean(all_calib_error)
	print('mean calib error=', mn)
	print('std calib error=', np.std(all_calib_error))
	return mn 



def adaptive_binning_algo(pred_label_pairs, beta, output_q_y_est=True): 
    """
    Inputs: 
    	pred_label_pairs: (list of tuples) 

    		Example: 
    			[(q_1, y_1), (q_2, y_2), ... (q_n, y_n)]
    		Where q is the predicted probability and y is the label 

    	beta : (int) target bin size (i.e. number of documents in a bin)

    Output: 
    	q_i_hat_list : (list of floats) each entry is the average predicted probabilities per bin 
    		length = number of bins

    	p_i_hat_list : (list of floats) each entry is the average empirical probabilites per bin 
    		length = number of bins 

    	calib_error : (float) The calibration error (root mean squared error per bin)
    	len_B_i     : (list of floats) each entry is the number of samples in a bin
    		length = number of bins

    """
    N = len(pred_label_pairs) 

    #Step 1: Sort pairs by prediction values qk in ascending order
    sorted_pred_label_pairs = sort_pairs(pred_label_pairs)
    
    #Step 2: For each, asign bin label 
    pair_label2bin_label = assign_bin_labels(sorted_pred_label_pairs, beta)
    
    #Step 3: Define bins 
    bin2pairs = define_bins(pair_label2bin_label, beta)
    
    #Step 4: Calculate the empirical and predicted probabilities per bin
    q_i_hat_list, p_i_hat_list = calc_per_bin_probs(bin2pairs, sorted_pred_label_pairs)
    
    #Step 5: Calculate calibration error 
    calib_error = calc_calib_error(q_i_hat_list, p_i_hat_list, bin2pairs, N)
    len_B_i = [len(bin2pairs[i+1]) for i in range(len(q_i_hat_list))]
    
    if output_q_y_est: 
    	return q_i_hat_list, p_i_hat_list, len_B_i, calib_error
    else: 
    	return calib_error, len_B_i
    
def calib_error_score(y_true, y_pred, NUM_BINS=10, output_q_y_est=False): 
	"""
	Scorer for sklearn
		See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring 
	"""
	pred_label_pairs = create_pred_label_pairs(y_pred, y_true)
	beta = np.round(len(y_true)/NUM_BINS)
	return adaptive_binning_algo(pred_label_pairs, beta, output_q_y_est)

def draw_samples_for_caliberation(NUM_BINS, NUM_SAMPLES, p_i_hat_list, q_i_hat_list, sigma_hat_list):
    '''Estimate caliberation error with 95% confidence interval'''
    CalibErr = []
    for s in range(NUM_SAMPLES):
        s_i_hat_list = []
        for j in range(NUM_BINS):
            p_s = np.random.normal(p_i_hat_list[j], np.sqrt(sigma_hat_list[j]))
            p_s = min(1, max(0, p_s))
            s_i_hat_list.append(p_s)
        calib_error, len_B_i = calib_error_score(s_i_hat_list, q_i_hat_list, NUM_BINS=NUM_BINS, output_q_y_est=False)
        CalibErr.append(calib_error)
    return CalibErr