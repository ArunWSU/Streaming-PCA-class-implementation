# -*- coding: utf-8 -*-
"""
Super class of Streaming PCA program, reads & stores input data
@author:Arun
"""

import os
import pandas as pd
import numpy  as np
from sklearn.decomposition import PCA


class DataReadPCA:
    # Initialize object with hist_train_data and current values
    def __init__(self,model_name,phase,train_data,retrain):
        self.model_name=model_name
        self.phase=phase
        self.hist_train_data=train_data
        self.hist_column_mean=np.mean(train_data,axis=0).values.reshape(1,-1)
        self.input_streams=self.hist_train_data.shape[1]
        self.time_step_elapsed=0
        self.retrain=retrain
        
    def timestep_update(self,time_step):
        if(self.time_step_elapsed==0):
            temp=self.projection_matrix[-1]
            self.reconst_train_val,self.reconst_diff,self.current_time_step,self.projection_matrix,self.subspace_train_calc,self.current_block_hist=([] for i in range(6))
            self.projection_matrix.append(temp)
            self.forecast_inp_hist=np.array(time_step).reshape(1,-1)
        else:
             self.forecast_inp_hist=np.vstack([self.forecast_inp_hist,time_step])
        self.current_time_step=self.forecast_inp_hist[-1]
        if(self.forecast_inp_hist.shape[0]-1 >= self.block_size_final): 
            self.current_block=self.forecast_inp_hist[-self.block_size_final+1:]
            self.current_block_hist.append(self.current_block)
            if(self.retrain):
                self.block_proj_train()
            self.proj_test(self.hist_column_mean)
        self.time_step_elapsed=self.time_step_elapsed+1
        
# PCA based forecasting of input data       
class BlockPCA(DataReadPCA):
    # Alternate flag is used for Block PCA update
    def __init__(BlockPCA,alternate_flag,size,model_name,phase,data,retrain):
        BlockPCA.block_size_max=size
        BlockPCA.alternate_flag=alternate_flag
        DataReadPCA.__init__(BlockPCA,model_name,phase,data,retrain)
        BlockPCA._pca_component_select()
        
        BlockPCA.current_time_step,BlockPCA.last_block, BlockPCA.covariance_matrix_, BlockPCA.projection_matrix, BlockPCA.subspace_train_calc, BlockPCA.explained_variance_ratio=([] for i in range(6))
        BlockPCA.reconst_train_val, BlockPCA.reconst_diff,BlockPCA.proj_reconst_diffmax=([] for i in range(3))
        
        # Multiple iterations blocks
        BlockPCA.proj_time,BlockPCA.proj_subspace,BlockPCA.proj_reconst=([] for i in range(3))
        
    # Selection of number of components
    # Check for maximum value in a column is within tolerance or perecentage of information/variance it holds
    def _pca_component_select(self):
        max_components=self.hist_train_data.shape[0]
        if(max_components > 10):
            max_components=10
        self.PCA_full=PCA(n_components=max_components)
        self.subspace_full=self.PCA_full.fit_transform(self.hist_train_data)
        
        for n in range(0,max_components):
         variance_captured=self.PCA_full.explained_variance_ratio_[:n].sum()*100
         if(variance_captured > 98):
            break
        
        self.no_components_=n
        self.PCA_train=PCA(n_components=n)
        self.subspace_train=self.PCA_train.fit_transform(self.hist_train_data)  
        self.reconstructed_train=self.PCA_train.inverse_transform(self.subspace_train)
        self.diff_train=self.hist_train_data-self.reconstructed_train
    
    # Determines the block size of m for train dataset
    def _pca_block_select(self):
        self.diff_block_sizes=np.arange(2,self.block_size_max,1)
        for self.current_block_size in self.diff_block_sizes:
                self.reconst_train_val,self.reconst_diff,self.forecast_block,self.projection_matrix=([] for i in range(4))
                self.n_loops=self.hist_train_data.shape[0]-self.current_block_size
                for i in np.arange(self.current_block_size,self.current_block_size+self.n_loops):
                        self.current_block=self.hist_train_data[-self.current_block_size+i:i].values
                        self.current_time_step.append(self.hist_train_data.iloc[i].values.reshape(1,-1))
                        self.block_proj_train()
                        self.proj_test(self.column_mean)
                        self.forecast_block.append(self.current_block)
                self.proj_time.append(np.array(self.projection_matrix).reshape(-1,self.input_streams))
                self.proj_subspace.append(np.array(self.subspace_train_calc).reshape(-1,1))
                self.proj_reconst.append(np.array(self.reconst_diff).reshape(-1,self.input_streams))
                self.proj_reconst_diffmax.append(self.proj_reconst[-1].max())
        self.block_size_final=np.argmin(np.array(self.proj_reconst_diffmax))+1
                
    
    def block_proj_train(self):
        self.column_mean=np.mean(self.current_block,axis=0).reshape(1,-1)
        self.recentered_last_block=self.current_block-self.column_mean
        
        if(self.alternate_flag):
            # Alternative online formulation
            self.z_t=self.recentered_last_block.sum(axis=0).values.reshape(-1,1)
            self.A_t=np.dot(np.transpose(self.recentered_last_block),self.recentered_last_block)
            self.covariance_matrix_.append((1/self.block_size-1)*(self.A_t-((1/self.block_size)*(self.z_t*np.transpose(self.z_t)))))
        else:
            # Numpy covariance function assumes row as variables and columns as realizations OPPOSITE of PCA input
            self.covariance_matrix_.append(np.cov(np.transpose(self.current_block)))
        
        u,self.sing_val,v=np.linalg.svd(self.covariance_matrix_[-1])
        self.projection_matrix.append(u[:,self.no_components_-1].astype('float64').reshape(-1,1))
        
         ## Check for correctness of variance ratio
        self.explained_variance_= self.sing_val[:self.no_components_]
        self.explained_variance_ratio.append(self.sing_val[:self.no_components_]/self.sing_val.sum())
        
     
    def proj_test(self,column_mean):
        self.recentered_val=self.current_time_step[-1]-column_mean
        self.subspace_train_calc.append(np.matmul(self.recentered_val,self.projection_matrix[-1]))
        self.reconst_recentered_val=np.matmul(self.subspace_train_calc[-1],np.transpose(self.projection_matrix[-1]))
        self.reconst_train_val.append(self.reconst_recentered_val+self.column_mean)
        self.reconst_diff.append(self.current_time_step[-1]-self.reconst_train_val[-1])
        self.reconst_max=np.amax(abs(self.reconst_diff[-1]))
#        if(abs(self.reconst_max) > 0.04):
#            self.violations.append(self.time_step_elapsed)
#            self.block_proj_train()
#            self.proj_test(self.hist_column_mean)
            
        
                       
    def pca_stream_forecast(self):
        self.last_block.append(self.hist_train_data[-self.block_size:].values)
        self.column_mean=np.mean(self.last_block[-1],axis=0).reshape(1,-1)
        self.recentered_last_block=self.last_block[-1]-self.column_mean
        
        if(self.alternate_flag):
            # Alternative online formulation
            self.z_t=self.recentered_last_block.sum(axis=0).values.reshape(-1,1)
            self.A_t=np.dot(np.transpose(self.recentered_last_block),self.recentered_last_block)
            self.covariance_matrix_.append((1/self.block_size-1)*(self.A_t-((1/self.block_size)*(self.z_t*np.transpose(self.z_t)))))
        else:
            # Numpy covariance function assumes row as variables and columns as realizations OPPOSITE of PCA input
            self.covariance_matrix_.append(np.cov(np.transpose(self.last_block[-1])))
        
        u,self.sing_val,v=np.linalg.svd(self.covariance_matrix_[-1])
        self.projection_matrix.append(u[:,self.no_components_-1].astype('float64').reshape(-1,1))
        
        ## Check for correctness of variance ratio
        self.explained_variance_= self.sing_val[:self.no_components_]
        self.explained_variance_ratio.append(self.sing_val[:self.no_components_]/self.sing_val.sum())
        self.sing_val_calc=np.sqrt(self.explained_variance_*(self.block_size-1))
        
        self.subspace_train_calc.append(np.matmul(self.recentered_last_block,self.projection_matrix[-1]))
        self.reconst_recentered_val=np.matmul(self.subspace_train_calc,np.transpose(self.projection_matrix[-1]))
        self.reconst_train_val.append(self.reconst_recentered_val+self.column_mean)
        self.reconst_diff.append(self.last_block[-1]-self.reconst_train_val[-1])
        self.proj_reconst_diffmax.append(self.reconst_diff[-1].max())
        