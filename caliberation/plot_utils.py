import matplotlib.pyplot as plt
import numpy as np

def graph_distributions(q, p):
    '''Distribution of P and Q for 4 classes'''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    assert len(q)==len(p), "Shape Mismatch"
    count = 0
    for i in range(2):
        for j in range(2):
            axs[i, j].hist([p[:, count], q[:, count]], bins=np.arange(0, 1, 0.1), rwidth=0.8, histtype='bar',\
                           stacked=False, align='mid', label=['true', 'predicted'])
            axs[i, j].axis(xmin=0, xmax=1, ymin=0, ymax=400)
            axs[i, j].set_xticks(np.arange(0, 1, 0.1))
            axs[i, j].set_xlabel('Prediction Strength')
            axs[i, j].set_ylabel('Count')
            axs[i, j].set_title('class : {} ({}) '.format(count, inv_class_dict[count]))
            axs[i, j].legend(loc='best')
            count+=1
    plt.show()
    return plt 

def graph_calibration_curve(q_i_hat_list, p_i_hat_list, caliberrors_mean, caliberrors_std, p_i_hat_std_list=None,\
                            draw_line=False):
    '''
    q_i_hat_list : Prediction Strength (dict type, q_i_hat_list[i]=list)
    p_i_hat_list : Empirical Frequency
    caliberrors_mean : Mean caliberation error
    caliberrors_std : std of caliberation error
    p_i_hat_std_list : std of empirical frequency estimates, used to draw error bars.
    draw_line : Whether to join points with a line
    
    '''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(9, 10))
    count = 0
    for i in range(2):
        for j in range(2):
            axs[i, j].scatter(p_i_hat_list[count], q_i_hat_list[count], color='g')
            if(draw_line):
                axs[i, j].plot(p_i_hat_list[count], q_i_hat_list[count])
            if(p_i_hat_std_list):
                axs[i, j].errorbar(p_i_hat_list[count], q_i_hat_list[count], xerr=0.0, \
                              yerr=p_i_hat_std_list[count], linestyle='')
            if(count!=1):
                axs[i, j].axis(xmin=0,xmax=1, ymin=0, ymax=1)
                axs[i, j].text(0.02, 0.9, r'CalibErr : {} $\pm$ {:.2e}'.format(np.round(caliberrors_mean[count], 4), \
                      caliberrors_std[count]), fontsize=12)
            else:
                axs[i, j].axis(xmin=0,xmax=0.4, ymin=0, ymax=0.4)
                axs[i, j].text(0.02, 0.36, r'CalibErr : {} $\pm$ {:.2e}'.format(np.round(caliberrors_mean[count], 4), \
                      caliberrors_std[count]), fontsize=12)
                
            axs[i, j].set_xlabel('Prediction Strength')
            axs[i, j].set_ylabel('Empirical Frequency')
            axs[i, j].set_title('class : {} ({}) '.format(count, inv_class_dict[count]))
            #plot y=x line
            lims = [
                np.min([axs[i, j].get_xlim(), axs[i, j].get_ylim()]),  # min of both axes
                np.max([axs[i, j].get_xlim(), axs[i, j].get_ylim()]),  # max of both axes
            ]
            axs[i, j].plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            axs[i, j].set_aspect('equal')
            count+=1
    plt.show()
    return plt 