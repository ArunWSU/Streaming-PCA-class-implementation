# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 07:47:37 2019

@author: WSU-PNNL
"""

import unittest,DataReadPCA_file
import pandas as pd
import numpy as np



class TestDataReadPCA(unittest.TestCase):
 
    
    def test_obj_creation(self):
        history=pd.DataFrame(data=np.array([[1,2],[3,4]]),columns=['test1','test2'])
        history.to_excel("D:/history_test.xlsx")
        self.hist1=DataReadPCA_file.DataReadPCA('history_test','D:/',0)
        self.assertIsInstance(self.hist1,DataReadPCA_file.DataReadPCA, msg="Object is not an instance of DataReadPCA class")
    
    def test_timestep_update(self):
        """ Test the timestep function with new additional data """
        history=pd.DataFrame(data=np.array([[1,2],[3,4]]),columns=['test1','test2'])
        history.to_excel("D:/history_test.xlsx")
        self.hist1=DataReadPCA_file.DataReadPCA('history_test','D:/',0)
        timestep_new=pd.DataFrame(data=np.array([5,6]).reshape(1,-1),columns=['test1','test2'])
        self.hist1.timestep_update(timestep_new)
        self.assertEqual(timestep_new,self.hist1.iloc[-1])
    
    if __name__ == '__main__':
        unittest.main()