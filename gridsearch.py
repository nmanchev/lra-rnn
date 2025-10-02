#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 06:40:28 2018

@author: manchev
"""
import numpy as np

import torch.nn as nn

from sklearn.model_selection import ParameterGrid

from tempOrder import TempOrderTask
from addition import AddTask
from permutation import PermTask
from tempOrder3bit import TempOrder3bitTask


from lra2_reg import run_experiment

if __name__=='__main__':
    
    task_name = "temp"
    seq_length = 20
    
    maxiter = 100000
    batch_size = 20
    chk_interval = 100
    n_hid = 100

    init   = nn.init.orthogonal_
    
    # GRID SEARCH
    param_grid = {"i_learning_rate" : [0.1, 0.01, 0.001],
                  "g_learning_rate" : [10, 1.0, 0.1, 0.01],
                  "k-steps"         : [10],
                  "c"               : [1.0],
                  "alpha"           : [2.0, 1.0, 0.1],
                  "gd_opt"          : ["SGD"],
                  "local_loss"      : ["LOG-PENALTY"]}

    grid = ParameterGrid(param_grid)
  
    grid_length = len(grid)
    grid_count = 1
    
    acc_hist   = []
    i_hist     = []
    g_hist     = []
    iter_hist  = []
    c_hist     = []
    k_hist     = []
    a_hist     = []

    for params in grid:
        print("Model %i of %i" % (grid_count, grid_length))
        if (len(acc_hist) > 0):
            print("Max acc. so far: %07.4f%%" % (max(acc_hist)*100))
            
        if grid_count<10:
          grid_count+=1
          continue;   
            
        i_lr = params["i_learning_rate"]
        g_lr = params["g_learning_rate"]           
        alpha = params["alpha"]
        c = params["c"]
        k_steps = params["k-steps"]
        gd_opt = params["gd_opt"]
        local_loss = params["local_loss"]
                
        val_acc, it = run_experiment(1234, init, task_name, n_hid, seq_length, batch_size, 
                                 gd_opt, i_lr, g_lr, k_steps, alpha, c, maxiter, chk_interval,
                                 local_loss, plot = False)
             
                
        iter_hist.append(it)
        acc_hist.append(val_acc)
        i_hist.append(i_lr)        
        g_hist.append(g_lr)
        c_hist.append(c)
        a_hist.append(alpha)
        k_hist.append(k_steps)
            
        grid_count+=1
    
    print("------------------- GRID SEARCH COMPLETED -------------------")
    
    best_idx = np.argmax(iter_hist)
    
    if (iter_hist[best_idx] >= maxiter):
        best_idx = np.argmax(acc_hist)
        
    acc_hist  = [i * 100 for i in acc_hist]
    print("Max accuracy    : %07.3f%%" % (acc_hist[best_idx]))
    print("i_learning_rate : %7.5f" % (i_hist[best_idx]))
    print("g_learning_rate : %7.5f" % (g_hist[best_idx]))
    print("C : %7.5f" % (c_hist[best_idx]))
    print("alpha : %7.5f" % (a_hist[best_idx]))
    print("k-steps : %7.5f" % (k_hist[best_idx]))
    
    all_stats = np.column_stack((iter_hist, i_hist, g_hist, c_hist, k_hist, a_hist, acc_hist))
    np.savetxt("all_stats.csv", all_stats, delimiter=",", header = "iter, i_lr, g_lr, C, k_steps, alpha, acc")
    
