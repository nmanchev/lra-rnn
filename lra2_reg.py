# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:33:51 2023

@author: Nick
"""

import sys
import torch

import time

import torch.nn as nn
import numpy as np

from tempOrder import TempOrderTask
from addition import AddTask
from permutation import PermTask
from tempOrder3bit import TempOrder3bitTask

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict

import matplotlib.pyplot as plt

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self,data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)
    
class SRNN(object):
    
    def __init__(self, task, seq_length, init, n_hid, batch_size, 
                 last_layer, glr, k_steps, alpha, grad_norm = 0.1, local_loss="MSE", err_norm = 0):
        
        super(SRNN, self).__init__()
                
        assert seq_length >= 10, "Sequence length must be at least 10."
        assert local_loss in ["MSE", "LOG-PENALTY"], "Unsupported local loss."
        
        self.dev = "cpu"        
        
        n_test_samples = 10000
        test_x, test_y = task.generate(n_test_samples, seq_length)
                
        self.X_test = Variable(torch.from_numpy(test_x).float().to(self.dev))        
        self.y_test = Variable(torch.from_numpy(test_y).float().to(self.dev))

        self.task = task               
        self.seq_length = seq_length
        self.input_size = task.nin
        self.n_hid = n_hid
        self.output_size = task.nout
        self.batch_size = batch_size
        self.glr = glr
        self.k_steps = k_steps
        self.c = grad_norm
        self.err_norm = err_norm
        self.alpha = alpha
        
        self.last_layer = last_layer
        
        if local_loss == "MSE":
            self.local_loss = self.mse
        elif local_loss == "LOG-PENALTY":
            self.local_loss = self.log_penalty
                
        self.activ = torch.tanh
        self.softmax = nn.Softmax(dim=1)
        self.mse = nn.MSELoss()
        
        self.h0 = torch.zeros(self.batch_size, self.n_hid, device=self.dev)
        
        self.Wxh = Parameter(init(torch.empty(self.input_size, self.n_hid, device=self.dev)))
        self.Whh = Parameter(init(torch.empty(self.n_hid, self.n_hid, device=self.dev)))
        self.Why = Parameter(init(torch.empty(self.n_hid, self.output_size, device=self.dev)))
        self.bh  = Parameter(torch.zeros(self.n_hid, device=self.dev))
        self.by  = Parameter(torch.zeros(self.output_size, device=self.dev))
        
        self.params = OrderedDict()
        self.params["Wxh"] = self.Wxh
        self.params["Whh"] = self.Whh
        self.params["Why"] = self.Why
        self.params["bh"] = self.bh
        self.params["by"] = self.by


    def normalize(self, a, force=False):
        if self.c != 0:
            norm_a = torch.linalg.norm(a, ord="fro")
            
            if force:
                # Indiscriminate normalization
                a = (self.c / torch.linalg.norm(a, ord="fro")) * a   
            elif norm_a >= self.c:
                # Only when |a|>=c
                a = (self.c / norm_a) * a
        
        return a

    
    def cross_entropy(self, pred, soft_targets):   
        return torch.mean(torch.sum(- soft_targets * torch.log(pred), 1))        

        
    def parameters(self):
        for key, value in self.params.items():
            yield value


    def _f(self, x, z):
        return z @ self.Whh + x @ self.Wxh + self.bh


    def forward(self, x, h0):
        h = []
        z = []        
        h.append(self._f(x[0, :, :], h0))
        z.append(self.activ(h[0]))
            
        for t in range(1, self.seq_length):

            h.append(self._f(x[t, :, :], z[t - 1]))
            z.append(self.activ(h[t]))

        out = z[-1] @ self.Why + self.by
        
        return h, z, out        


    def log_penalty(self, z, y):
        loss = torch.mean(torch.log(1 + (y-z)**2)) 
        return loss

        
    def mse(self, pred, target):
        loss = self.mse(pred,target)
        return loss        

        
    def global_loss(self, out, y):
        if self.last_layer == "lastSoftmax": 
            out = self.softmax(out)           
            return self.cross_entropy(out, y)                            
        elif self.last_layer == "lastLinear":
            return  self.mse(out, y).sum()            
        else:
            raise Exception("Unsupported classification type.")
        
        
    def validate(self):
        
        with torch.no_grad():
                
            batch_size = 1000           
            n_batches = self.X_test.shape[1] // batch_size
            
            total_loss = np.zeros(n_batches)
            total_err = np.zeros(n_batches)
            
            for batch_idx in range(n_batches):
                                
                batch_start = batch_idx*batch_size
                
                X = self.X_test[:,batch_start:batch_start + batch_size,:]
                y = self.y_test[batch_start:batch_start + batch_size]
                
                h0 = self._f(X[0, :, :], torch.zeros(batch_size, self.n_hid, device=self.dev))
                _, _, out = self.forward(X, h0)

                total_loss[batch_idx] = self.global_loss(out, y)

                if self.last_layer == "lastSoftmax":                                
                    out = torch.argmax(out, axis=1)
                    y = torch.argmax(y, axis=1)                                    
                    total_err[batch_idx] = (~torch.eq(out, y)).float().mean()                    
                elif self.last_layer == "lastLinear":
                    total_err[batch_idx] = (((y - out) ** 2).sum(axis=1) > 0.04).float().mean()                    
                else:
                    raise Exception("Unsupported classification type.")

        return total_loss.mean(), total_err.mean()


    def get_target(self, h_l_m1, z_l, x, loss):
                
        h_l_m1_temp = h_l_m1
        z_l_hat = z_l
        
        for k in range(0, self.k_steps):
            dh_l_m1 = torch.autograd.grad(loss, h_l_m1_temp, retain_graph=True)[0]

            if self.alpha != 0:
                omega = self.regularize(loss, h_l_m1_temp, z_l_hat, self.Whh)                
                dh_l_m1 = dh_l_m1 + self.alpha * torch.autograd.grad(omega, h_l_m1_temp)[0]
                                    
            dh_l_m1 = self.normalize(dh_l_m1)
                 
            h_l_m1_hat = h_l_m1_temp - self.glr * dh_l_m1
            z_l_m1_hat = self.activ(h_l_m1_hat)
            h_l_hat = z_l_m1_hat@ self.Whh + x@self.Wxh + self.bh
            z_l_hat = self.activ(h_l_hat)        
            h_l_m1_temp = h_l_m1_hat
            loss = self.local_loss(z_l, z_l_hat)
            
        return z_l_m1_hat, loss


    def regularize(self, E, h_t, h_tp1, W):
                
        dEdh_tp1 = torch.autograd.grad(E, h_tp1,retain_graph=True)[0]
        d_tanh = torch.diag(1 - (torch.tanh(h_t)**2))
        
        term_1 = (dEdh_tp1@W.T).T*d_tanh
        
        omega = (torch.norm(term_1) / torch.norm(dEdh_tp1) - 1) ** 2
        
        return omega


    def calc_grads(self, x, y, h, z, out):
        losses = np.zeros(self.seq_length)
        
        # Calculate dWhy, dby
        global_loss = self.global_loss(out, y)
        
        #losses[-1] = loss
        dWhy = torch.autograd.grad(global_loss, self.Why, retain_graph=True)[0]
        dby = torch.autograd.grad(global_loss, self.by, retain_graph=True)[0]
        
        # Last layer                
        h_max_temp = h[-1]        
        h_out_hat = out
                
        if self.alpha != 0:
            omega = self.regularize(global_loss, h[-1].detach(), h_out_hat, self.Why)            
            dWhy = dWhy + self.alpha * torch.autograd.grad(omega, self.Why)[0]
        
        dWhy = self.normalize(dWhy)
                        
        for k in range(0, self.k_steps):
            dh_max = torch.autograd.grad(global_loss, h_max_temp, retain_graph=True)[0]
            
            if self.alpha != 0:
                omega = self.regularize(global_loss, h_max_temp, h_out_hat, self.Why)
                dh_max = dh_max + self.alpha * torch.autograd.grad(omega, h_max_temp)[0]
            
            dh_max = self.normalize(dh_max)

            h_max_hat = h_max_temp - self.glr * dh_max
            h_max_temp = h_max_hat
            z_max_hat = self.activ(h_max_hat)
            h_out_hat = z_max_hat@self.Why + self.by
            
            global_loss  = self.global_loss(h_out_hat, y)

        dWhh = torch.zeros((self.seq_length, self.n_hid, self.n_hid))
        dbh = torch.zeros((self.seq_length, self.n_hid))
        dWxh = torch.zeros((self.seq_length, self.input_size, self.n_hid))
        z_hat =  [None] * self.seq_length
        z_hat[-1] = z_max_hat
        
        loss = self.local_loss(z[-1], z_hat[-1])        
        
        for l in range(self.seq_length - 1, 0, -1):
            
            losses[l] = loss
                        
            # if the local loss for the layer is too small maybe skip it altogether?
            #if loss < 0.00001:
            #    continue
            
            dWhh[l] = torch.autograd.grad(loss, self.Whh, retain_graph=True)[0]                        
            dbh[l] = torch.autograd.grad(loss, self.bh, retain_graph=True)[0]
            dWxh[l] = torch.autograd.grad(loss, self.Wxh, retain_graph=True)[0]
            

            # Check if loss <> 0, otherwise omega -> NaN
            if (self.alpha != 0) and (l<len(h)-1) and (loss != 0):
                omega = self.regularize(loss, h[l], h[l+1], self.Whh)
                #dWhh[l] = dWhh[l] + self.alpha * torch.autograd.grad(omega, self.Whh, retain_graph=True)[0]
                dWxh[l] = dWxh[l] + self.alpha * torch.autograd.grad(omega, self.Wxh, retain_graph=True)[0]

            
            dWhh[l] = self.normalize(dWhh[l])
            dWxh[l] = self.normalize(dWxh[l])            
            
            z_hat[l-1], loss = self.get_target(h[l-1], z[l], x[l, :, :], loss)

        return global_loss.item(), losses, dWxh, dWhy, dby, dWhh, dbh
                

    def train(self, opt, maxiter, chk_iter, calc_dWhh_norm = True):
        
        training = True
        
        losses = []
        global_losses = []
        dWhh_norms = []
        dWxh_norms = []
        
        patience = 0
        max_patience = 30
        
        test_loss, lowest_err = self.validate()
        
        print("Starting val loss: {:.4f}\t Starting val accuracy: {:.4%}\n".format(test_loss, 1-lowest_err))

        it = 1
        
        while (training and (it <= maxiter)):
            
            X, y = self.task.generate(self.batch_size, self.seq_length)
            X = Variable(torch.from_numpy(X))
            y = Variable(torch.from_numpy(y))

            opt.zero_grad()
                                    
            h, z, out = self.forward(X, self.h0)        
            global_loss, train_loss, dWxh, dWhy, dby, dWhh, dbh = self.calc_grads(X, y, h, z, out)
            global_losses.append(global_loss)
            losses.append(train_loss)
                        
            if calc_dWhh_norm:
                dWhh_norms.append(torch.linalg.matrix_norm(dWhh).numpy())
                dWxh_norms.append(torch.linalg.matrix_norm(dWxh).numpy())

            if np.isnan(train_loss[-1]):
                print("Training loss is NAN. Val acc: {:.4f} Aborting...".format(1-lowest_err))
                return global_losses, losses, dWhh_norms, dWxh_norms, 1-lowest_err, it
            
            self.Whh.grad = dWhh.sum(0)
            self.Wxh.grad = dWxh.sum(0)
            self.bh.grad = dbh.sum(0)
            self.Why.grad = dWhy            
            self.by.grad = dby
            
            opt.step()
            
            if (it % chk_iter == 0):
                
                patience += 1                
                valid_loss, valid_err = self.validate()
                
                if (valid_err < lowest_err):
                    patience = 0
                    lowest_err = valid_err
                
                print_str = "It {:,}/{:,}\t\tTrain loss: {:.4f}\t\tVal loss: {:.4f}\tVal Acc: {:.4%}\tHighest: {:.4%} {}".format(it, maxiter, global_loss, valid_loss, 1-valid_err, 1-lowest_err, patience)
                
                print(print_str)

                if valid_err < 0.0001:
                    print("PROBLEM SOLVED.")
                    training = False

                if patience >= max_patience:
                    print("Validation error not reducing. Val acc: {:.4f} Aborting...".format(1-lowest_err))
                    return global_losses, losses, dWhh_norms, dWxh_norms, 1-lowest_err, it

            it+=1
    
        return global_losses, losses, dWhh_norms, dWxh_norms, 1-lowest_err, it
    
def plot_losses(global_losses, losses, dWhh_norms, dWxh_norms): 

    losses = np.array(losses)
    dWhh_norms = np.array(dWhh_norms)
    dWxh_norms = np.array(dWxh_norms)
    
    steps = losses.shape[0]
    n_layers = losses.shape[1]
        
    fig, axs = plt.subplots(3, n_layers+1, figsize=(20,10), sharey=False)
    
    x = np.arange(1, steps + 1)        
    
    for layer in range(n_layers):
        axs[0,layer].bar(x, losses[:, layer])        
        axs[0,layer].get_xaxis().set_ticks([])
        axs[0,layer].set_xlabel("L" + str(layer))
        
        axs[1,layer].bar(x, dWhh_norms[:, layer])        
        axs[1,layer].get_xaxis().set_ticks([])
        
        axs[2,layer].bar(x, dWxh_norms[:, layer])        
        axs[2,layer].get_xaxis().set_ticks([])
            
    axs[0,layer+1].bar(x, global_losses)        
    axs[0,layer+1].get_xaxis().set_ticks([])
    axs[0,layer+1].set_xlabel("Global Loss")
        
    axs[1,layer+1].set_visible(False)
    axs[2,layer+1].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    
def run_experiment(seed, init, task_name, n_hid, seq, batch_size, opt, lr, glr, 
                   k_steps, alpha, grad_norm, maxiter, chk_iter, local_loss,
                   plot = False):
    
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    if (task_name == "perm"):        
        task = PermTask(rng, "float32")
    elif (task_name == "add"):
        task = AddTask(rng, "float32")
    elif (task_name == "temp"):
        task = TempOrderTask(rng, "float32")
    elif (task_name == "temp3bit"):
        task = TempOrder3bitTask(rng, "float32")     
    else:
        print("Unknown task {}. Aborting...".format(task_name))
        return

    # Print all parameters (for logging purposes)
    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - RNN LRA")
    print("******************************************************")    
    print("task            : %s" % task_name)
    print("sequence length : %i" % seq)
    print("n_hid           : %i" % n_hid)
    print("learning_rate   : %f" % lr)
    print("g_learning_rate : %f" % glr)
    print("k_steps         : %d" % k_steps)
    print("c               : %f" % grad_norm)
    print("alpha           : %f" % alpha)
    print("optimization    : %s" % opt)
    print("-----------------")    

    if task.classifType == "lastSoftmax":
        print("global loss     : cross-entropy")
    elif task.classifType == "lastLinear":
        print("global loss     : MSE")
    else:
      raise Exception("Unsupported classification type.")
      
    if local_loss == "MSE":
        print("local loss      : MSE ")
    elif local_loss == "LOG-PENALTY":
        print("global loss     : LOG-PENALTY")
    else:
      raise Exception("Unsupported local loss.")

    print("-----------------")
    print("init            : %s" % str(init).split(" ")[1])
    print("maxiter         : %i" % maxiter)
    print("batch_size      : %i" % batch_size)
    print("chk_interval    : %i" % chk_iter)
    print("******************************************************")
        
    init = nn.init.orthogonal_
                
    model = SRNN(task, seq, init, n_hid, batch_size, task.classifType, glr, k_steps, alpha, grad_norm, local_loss)

    if opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, nesterov=False)
    elif opt == "Nesterov":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif opt == "RMS":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        print("Unknown optimiser %s. Aborting..." % opt)
        return

    global_losses, losses, dWhh_norms, dWxh_norms, val_acc, it = model.train(optimizer, maxiter, chk_iter, calc_dWhh_norm = plot)
    
    if plot:
        plot_losses(global_losses, losses, dWhh_norms, dWxh_norms)
        
    return val_acc, it

def main(args): 
    
    # Set a random seed for reproducibility
    seed = 1234
    
    # Train the network
    start_time = time.time()
    
    task = "perm"
    init   = nn.init.orthogonal_
    n_hid = 100
    seq = 20
    batch_size = 20
    opt = "SGD"
    ilr = 0.1
    glr = 10.0
    k_steps = 10
    c = 1.0
    maxiter = 100000
    chk_interval = 100
    local_loss = "LOG-PENALTY"
    alpha = 0.1
    
    val_acc, it = run_experiment(seed, init, task, n_hid, seq, batch_size, opt, ilr, glr, 
                   k_steps, alpha, c, maxiter, chk_interval, local_loss, plot=True)
    
    print("Elapsed time: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    
    
if __name__=='__main__':    

    main(sys.argv)
    