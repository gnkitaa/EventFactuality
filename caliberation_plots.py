import sys, os
sys.path.append("./caliberation/")
from adaptive_binning_calib_error import *
from plot_utils import *

def do_caliberation(q, y):
    
    if not os.path.exists("./Figures"):
        os.makedirs("./Figures")
    
    # Caliberation
    class_dict = {'positive':0, 'negative':1, 'uncommitted':2, 'not_applicable':3}
    inv_class_dict = {} 
    for k,v in class_dict.items():
        inv_class_dict[v] = k
        
    plt = graph_distributions(q, y, inv_class_dict)
    plt.savefig("./Figures/distributions.jpeg")
    
    
       
    NUM_BINS = 30
    Q_hat_list, P_hat_list, calib_errors, len_B_list  = {},{},{},{}
    for i in range(4):
        q_i_hat_list, p_i_hat_list, len_B_i, calib_error = calib_error_score(y[:,i], \
                                                           q[:,i], NUM_BINS=NUM_BINS, \
                                                           output_q_y_est=True)
        calib_errors[i] = calib_error
        Q_hat_list[i] = np.array(q_i_hat_list)
        P_hat_list[i] = np.array(p_i_hat_list)
        len_B_list[i] = np.array(len_B_i)

    NUM_BINS_FORMED = len(Q_hat_list[0])

    sigma_hat_list = {}
    P_hat_std_list = {}
    for i in range(4):
        sigma_hat_list[i] = np.zeros(NUM_BINS_FORMED)
        P_hat_std_list[i] = np.zeros(NUM_BINS_FORMED)
        for j in range(NUM_BINS_FORMED):
            sigma_hat_list[i][j] = P_hat_list[i][j]*(1-P_hat_list[i][j])/len_B_list[i][j]
            P_hat_std_list[i][j] = np.sqrt(sigma_hat_list[i][j])


    CalibStd = {}
    CalibMean = {}
    for i in range(4):
        CalibErr = draw_samples_for_caliberation(NUM_BINS_FORMED, 1000, \
                                                 P_hat_list[i], Q_hat_list[i], sigma_hat_list[i])
        CalibStd[i] = 1.96*np.std(CalibErr)
        CalibMean[i]= np.mean(CalibErr)

    myplt = graph_calibration_curve(Q_hat_list, P_hat_list, CalibMean, CalibStd, inv_class_dict)
    myplt.savefig("./Figures/caliberation_plots.jpeg")
    
    plt.figure()
    plt.bar(np.arange(4), [CalibMean[i] for i in range(4)])
    plt.errorbar(np.arange(4), [CalibMean[i] for i in range(4)], [CalibStd[i] for i in range(4)],\
                 linestyle='', color='k')
    plt.xticks(np.arange(4))
    plt.xlabel('Class')
    plt.ylabel('Caliberation error')
    plt.savefig("./Figures/caliberation_errors_histogram.jpeg")