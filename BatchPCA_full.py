# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:38:18 2019

@author: WSU-PNNL
"""

#%% Reading PCA input data
# Variables to set phase(0,1,2)
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
#matplotlib inline

#import matplotlib
#matplotlib.use('Agg')

# Visualizing line plots
def gen_line_plot(x,y,label_x,label_y,visualize_fig,save_fig,fig_name,file_name,file_path):
    if(visualize_fig):
        plt.figure()
        plt.plot(x,y,linewidth=1, marker='o',markersize=3)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.tight_layout()
        plt.show()
        if(save_fig==1):
            plt.savefig(os.path.join(file_path,fig_name+file_name+'.png'),dpi=300)
#            plt.savefig(os.path.join(file_path,fig_name+file_name+'.eps'),format='eps',dpi=600)
if __name__=='__main__':    
    phase=0
    visualize_fig=1
    save_fig=1
    
    file_name='13bussecondaryrandom'
    file_path='D:\PCAdatasets'
    file=os.path.join(file_path,file_name)
    volt_phase_full=pd.read_excel((file+'.xlsx'),phase)
    #drop_bus='sourcebus.'+str(phase+1)
    #volt_phase_full.drop(columns=[drop_bus],inplace=True)
    bus_names=list(volt_phase_full.columns)
    #%% Determine No of principal components 
    i=10
    PCA_full_check=PCA(n_components=i)
    subspace_full_check=PCA_full_check.fit_transform(volt_phase_full)
    var=sum(PCA_full_check.explained_variance_ratio_[0:2])
    # Check for maximum column is within tolerance
    for n in range(0,i):
     if(max(abs(subspace_full_check[:,n])) < 0.01):
        break
    #%% Subspace projection of full dataset
    PCA_full=PCA(n_components=n)
    subspace_full=PCA_full.fit_transform(volt_phase_full)
    file_name=file_name+' components_'+str(n)+' phase '+str(phase)
    reconstructed_full=PCA_full.inverse_transform(subspace_full)
    reconstructed_diff=reconstructed_full-volt_phase_full
    select_bus=bus_names[1]
    
    x=np.arange(0,volt_phase_full.shape[0])
    #file_path=(r'D:\Image check')
    file_path=(r'C:\Users\WSU-PNNL\OneDrive - Washington State University (email.wsu.edu)\Programs\MATLAB programs\PCA research project\PCA plots and results\Applying PCA\Streaming PCA comparison results\13 bus secondary higher res full day')
    comp=PCA_full.components_
    comp_size=np.arange(0,comp.shape[1])
    gen_line_plot(comp_size,np.transpose(comp),'Different nodes along feeder','PCA component values',visualize_fig,save_fig,'PCA projection matrix',file_name,file_path)
    gen_line_plot(x,subspace_full,'Time step','Sub space full values',visualize_fig,save_fig,'Subspace projection full',file_name,file_path)
    gen_line_plot(x,reconstructed_diff,'Time step','Reconstructed full difference for each node',visualize_fig,save_fig,'Reconstructed full difference',file_name,file_path)
     #%% Subspace projection of Train and test dataset
     # Effect of size of input dataset on algorithm
    train_percent=0.7
    hist_end_index=int(volt_phase_full.shape[0]*train_percent)
    train_data=volt_phase_full[0:hist_end_index].copy()
    test_data=volt_phase_full[hist_end_index:].copy()
    
    PCA_train=PCA(n_components=n)
    subspace_train=PCA_train.fit_transform(train_data)
    reconstructed_train=PCA_train.inverse_transform(subspace_train)
    train_diff=train_data-reconstructed_train
    x=np.arange(0,train_data.shape[0])
    gen_line_plot(x,subspace_train,'Time step','Sub space train values',visualize_fig,save_fig,'Subspace train',file_name,file_path)
    gen_line_plot(x,train_diff,'Time step','Reconstructed train difference for each node',visualize_fig,save_fig,'Reconstructed train',file_name,file_path)
    
    subspace_test=PCA_train.fit_transform(test_data)
    reconstructed_test=PCA_train.inverse_transform(subspace_test)
    test_diff=test_data-reconstructed_test
    x=np.arange(0,test_data.shape[0])
    gen_line_plot(x,subspace_test,'Time step','Sub space test values',visualize_fig,save_fig,'Subspace projection test',file_name,file_path)
    gen_line_plot(x,test_diff,'Time step','Reconstructed test difference for each node',visualize_fig,save_fig,'Reconstructed test',file_name,file_path)
    
    max_reconstructed_full=reconstructed_diff.values.max()
    max_reconstructed_train=train_diff.values.max()
    max_reconstructed_test=test_diff.values.max()
    
    x=np.arange(0,volt_phase_full.shape[0])
    gen_line_plot(x,volt_phase_full,'Time step','Voltage(pu)',visualize_fig,save_fig,'Column plot of matrix',file_name,file_path)
    x=np.arange(0,volt_phase_full.shape[1])
    gen_line_plot(x,volt_phase_full.T,'Different nodes of a phase of feeder','Voltage(pu)',visualize_fig,save_fig,'Row plot of matrix',file_name,file_path)
