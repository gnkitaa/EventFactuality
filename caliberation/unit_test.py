"""
"Adaptive Binning" algorithm by Nguyen and O'Connor 
(Algorithm 1)  https://arxiv.org/pdf/1508.05154.pdf

Author: Katie Keith
2020-09-13
"""
import numpy as np
from collections import defaultdict, Counter 
from adaptive_binning_calib_error import *

def test_sort_pairs(): 
    pred_label_pairs = [(0.9, 0.6), (0.5, 0.9), (0.6, 0.1)]
    assert sort_pairs(pred_label_pairs) == [(0.5, 0.9), (0.6, 0.1), (0.9, 0.6)]
    
def test_assign_bin_labels(): 
    sorted_pred_label_pairs = [(1, 1), (1, 1), (1, 1), (1, 1)]
    beta = 2 
    print(assign_bin_labels(sorted_pred_label_pairs, beta) )
    assert assign_bin_labels(sorted_pred_label_pairs, beta) == {0: 1, 1: 1, 2: 2, 3: 2}
    
def test_define_bins(): 
    pair_label2bin_label = {0: 1, 1: 1, 2: 2, 3: 2, 4:3}
    bin2pairs = {1: [0, 1], 2: [2, 3, 4]}
    beta = 2
    print(define_bins(pair_label2bin_label, beta))
    assert define_bins(pair_label2bin_label, beta) == bin2pairs
    
    
def test_calc_per_bin_probs(): 
    bin2pairs = {1: [0, 1], 2: [2, 3, 4]}
    sorted_pred_label_pairs = [(0.5, 0.6), (0.6, 0.7), (0.8, 0.7), (0.1, 0.1), (0.2, 0.3)]
    q_i_hat_list = [0.55, 0.3666666]
    p_i_hat_list = [0.65, 0.3666666]
    out1, out2 = calc_per_bin_probs(bin2pairs, sorted_pred_label_pairs)
    
    import numpy.testing as npt
    npt.assert_almost_equal(np.array(out1), np.array(q_i_hat_list))
    npt.assert_almost_equal(np.array(out2), np.array(p_i_hat_list))
    
def unit_tests(): 
	test_sort_pairs()
	test_assign_bin_labels()
	test_define_bins()
	test_calc_per_bin_probs()
    
if __name__ == '__main__':
    unit_tests()