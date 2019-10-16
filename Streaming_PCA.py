# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:51:17 2019

@author: WSU-PNNL
"""

from DataReadPCA_file import DataReadPCA,BlockPCA
from  BatchPCA_full import gen_line_plot
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt 

file_name='13bussecondaryrandom'
file_path='D:\PCAdatasets'
phase=2

file_to_read=os.path.join(file_path,file_name+'.xlsx')
volt_phase_full=pd.read_excel(file_to_read,phase) # Sheet number denotes phase
hist_index=int(volt_phase_full.shape[0]*0.7)
#%% Historical Training
max_block_size=12
retrain=0
stream_obj=BlockPCA(0,max_block_size,file_name,phase,volt_phase_full[0:hist_index],retrain)
stream_obj._pca_block_select()
#%% Visualize history
visualize_fig=0
save_fig=0
file_path=(r'C:\Users\WSU-PNNL\OneDrive - Washington State University (email.wsu.edu)\Programs\MATLAB programs\PCA research project\PCA plots and results\Applying PCA\Streaming PCA comparison results\Block PCA retraining')
x=np.arange(1,stream_obj.proj_time[0].shape[0]+1)   
gen_line_plot(x,stream_obj.proj_time[0],'Different nodes along feeder','PCA component values',visualize_fig,save_fig,'PCA projection matrix'+'block_size'+str(max_block_size),file_name,file_path)
x=np.arange(1,stream_obj.proj_subspace[0].shape[0]+1)
gen_line_plot(x,stream_obj.proj_subspace[0],'Time step','Sub-space full values',visualize_fig,save_fig,'Subspace projection full'+'block_size'+str(max_block_size),file_name,file_path)
gen_line_plot(x,stream_obj.proj_reconst[0],'Time step','Residuals for each node',visualize_fig,save_fig,'Reconstructed full difference'+'block_size'+str(max_block_size),file_name,file_path)
x=np.arange(1,len(stream_obj.proj_reconst_diffmax)+1)
gen_line_plot(x,np.array(stream_obj.proj_reconst_diffmax).reshape(-1,1),'Block size','Maximum residual',visualize_fig,save_fig,'Max residual'+'block_size'+str(max_block_size),file_name,file_path)
#%% Testing
#stream_obj.block_size_final=7
test_data=volt_phase_full[hist_index:]
test_data.reset_index(inplace=True,drop=True)
for row in test_data.itertuples(index=False):
   stream_obj.timestep_update(row)
#%% Visualize Forecast
visualize_fig=1
save_fig=0
add='block_size'+str(max_block_size)+'phase'+str(phase)+'No retraining'

subspace_test=np.array(stream_obj.subspace_train_calc).reshape(-1,1)
residual_test=np.array(stream_obj.reconst_diff).reshape(-1,stream_obj.input_streams)
forecast_test=np.array(stream_obj.reconst_train_val).reshape(-1,stream_obj.input_streams)
mat_diff_residual=stream_obj.forecast_inp_hist[stream_obj.block_size_final:]-forecast_test

x=np.arange(1,subspace_test.shape[0]+1)
gen_line_plot(x,subspace_test,'Time step','Sub-space test values',visualize_fig,save_fig,'Subspace test'+add,file_name,file_path)
gen_line_plot(x,residual_test,'Time step','Residual difference for each node',visualize_fig,save_fig,'Reconstructed full difference'+add,file_name,file_path)
#%% Multi-line single image plot
#visualize_fig=0
#save_fig=0
#diff=[]
#for i in range(len(stream_obj.proj_time)-1):
#    diff.append(stream_obj.proj_time[i][1:,:]-stream_obj.proj_time[i+1])
#    
#max_diff=[np.amax(i,axis=1) for i in diff]
#
#x=np.arange(1,len(max_diff[-1])+1)
#if(visualize_fig):
#    plt.figure()
#    for i in range(1,len(max_diff)+1):
#            y=max_diff[-i][i-1:]
#            plt.plot(x,y,linewidth=1, marker='o',markersize=3)
#    plt.xlabel('Time steps')
#    plt.ylabel('Projection Matrix difference(Scale 0-1)') 
#    plt.tight_layout()
#    plt.show()   
#    if(save_fig):
#        plt.savefig(os.path.join(file_path,'Proj_diff'+file_name+'block_size'+str(max_block_size)+'.png'),dpi=400)
#
## Multiple image plot       
#idx=[1,3,5]    
#for i in idx:
#        y=max_diff[-i]
#        x=np.arange(1,y.shape[0]+1,1)
#        if(visualize_fig):
#            plt.figure()
#            plt.plot(x,y,linewidth=1, marker='o',markersize=3)
#            plt.xlabel('Time steps')
#            plt.ylabel('Projection Matrix difference(Scale 0-1)') 
#            plt.tight_layout()
#            plt.show()
#            if(save_fig):
#                plt.savefig(os.path.join(file_path,'Proj_diff'+file_name+'block_size'+str(max_block_size)+'from last'+str(-i)+'.png'),dpi=400)     
#%% Heat map for visualization
#import matplotlib as mpl
#
#fig=plt.figure()
#
#a=diff[-1][0:200,:]
## Color map of fixed colors
#cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['blue','black','red'],256)
#bounds=[0,1]
#norm = mpl.colors.BoundaryNorm(bounds, cmap2.N)                                       
#
## tell imshow about color map so that only set colors are used
#img = plt.imshow(a,interpolation='nearest', cmap = cmap2,origin='lower')
#
## make a color bar
#plt.colorbar(img,cmap=cmap2)
#plt.grid(True,color='white')
#plt.show()
        
#%% Comments
#projection_matrix,projection_matrix_diff,avg_explained_ratio=([] for i in range(3))
#for block_size in range(2,3):
#    history=volt_phase_full[0:block_size]
#    stream_obj=BlockPCA(0,block_size,model_name,phase,history)
#    for time_step in range(block_size,int(volt_phase_full.shape[0]*0.7)):
#        stream_obj.timestep_update(volt_phase_full.iloc[time_step])
#        stream_obj.pca_stream_forecast()
#    
#    #Saving results
#    projection_matrix.append(np.array(stream_obj.projection_matrix).reshape(-1,13))
#    avg_explained_ratio.append(np.array(stream_obj.explained_variance_ratio).mean())
#        
        
# =============================================================================
# stream_obj=BlockPCA(0,5,model_name,phase,history)
# 
# # Training data
# train_end_index=int(volt_phase_full.shape[0]*0.7)
# history=volt_phase_full[0:train_end_index]
# 
# # Streaming update of data
# for i in range(train_end_index,volt_phase_full.shape[0]):
#     stream_obj.timestep_update(volt_phase_full.iloc[i])
#     stream_obj.pca_stream_forecast()
# 
# =============================================================================
# Visualization of test data
# =============================================================================
# save_fig=1
# file_path=(r'C:\Users\WSU-PNNL\OneDrive - Washington State University (email.wsu.edu)\Programs\MATLAB programs\PCA research project\PCA plots and results\Applying PCA\Streaming PCA comparison results')
# subspace_test=PCA_train.fit_transform(test_data)
# reconstructed_test=PCA_train.inverse_transform(subspace_test)
# test_diff=test_data-reconstructed_test
# x=np.arange(0,test_data.shape[0])
# gen_line_plot(x,subspace_test,'Time step','Sub space test values',visualize_fig,save_fig,'Subspace projection test',file_name,file_path)
# gen_line_plot(x,test_diff,'Time step','Reconstructed test difference for each node',visualize_fig,save_fig,'Reconstructed test',file_name,file_path)
# =============================================================================
