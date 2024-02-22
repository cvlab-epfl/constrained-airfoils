#!/usr/bin/env python3
#%%---------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os,pickle,gzip,sys,logging

import torch
#-----------------------------------------------------------------------------
#                                 LOGGING
#-----------------------------------------------------------------------------

def log_config_default(log_name):
	# write INFO and below to stdout
	handler_stdout = logging.StreamHandler(sys.stdout)
	handler_stdout.setFormatter(logging.Formatter(fmt='{message}', style='{'))
	handler_stdout.addFilter(lambda r: r.levelno <= logging.INFO)
	
	# write WARNING and above to stderr, resulting in red text in jupyter
	handler_stderr = logging.StreamHandler(sys.stderr)
	handler_stderr.setFormatter(logging.Formatter(fmt='{levelname:<7}|{filename}:{lineno}| {message}', style='{'))
	handler_stderr.setLevel(logging.WARNING)
	
	# we don't use the root log, because all libraries are writing trash to it
	# instead all our code uses the 'exp' for "experiment" log space
	log = logging.getLogger(log_name)
	log.setLevel(logging.DEBUG)
	for h in [handler_stdout, handler_stderr]: log.addHandler(h)
	return log

logf = log_config_default('exp')


def setLogFile(filepath,log_obj = logf,newP=False):
    
    if(newP):
        removeFile(filepath,verbP=False)
    
    handler_file = logging.FileHandler(filepath)
    handler_file.setFormatter(logging.Formatter(fmt = '{asctime}|{levelname:<7}| {message}', style = '{', datefmt = '%m-%d %H:%M:%S'))	
	# add handlers to root
    log_obj.addHandler(handler_file)
    log_obj.info(f'Log file {filepath} initialized')
    
def removeFile(fileName,verbP=True):
    if(os.path.isfile(fileName)):
        try:
            os.remove(fileName)
        except IOError:
            if(verbP):
                print('removeFile: cannot remove', fileName)
                
                
#------------------------------------------------------------------------------
#                               FILES
#------------------------------------------------------------------------------

def loadFromFile(fileName,gzipP=False,verbP=True):
    if(fileName is not None):
        try:
            if(gzipP):
                with gzip.open(fileName+'.gz', 'rb') as f:
                  return(pickle.load(f))
            else:
                with open(fileName,'rb') as f:
                    return(pickle.load(f))
        except IOError:
            if(verbP):
                print('loadFromFile: cannot open', fileName)
            return None

def dumpToFile(fileName,obj,verbP=True):
    if(fileName is not None):
        try:
            with open(fileName,'wb') as f:
                pickle.dump(obj,f)
        except IOError:
            if(verbP):
                print('dumpToFile: cannot open', fileName)  
        
def pytDir(dirName=None,newP=False):
    
    defaultDir=os.path.join(os.environ['HOME'],'code/pyt')
    
    if(dirName is None):
        return(defaultDir)
    elif(os.path.isdir(dirName)):
        return dirName
    else:
         dirName=os.path.join(defaultDir,dirName)
         if(os.path.isdir(dirName)):
             return dirName
         elif(newP):
             os.mkdir(dirName)
             return(dirName)
    return(None)
#------------------------------------------------------------------------------
#                               TENSOR
#------------------------------------------------------------------------------

def makeTensor(x,floatP=True,gradP=False):
    
    if(torch.is_tensor(x)):
        x = x.clone().detach()
        x = x.to(currentDevice())
    else:
        if(floatP):
            x = np.asarray(x,dtype=np.float32)
        else:
            x = np.asarray(x,dtype=np.int32)
        x = torch.tensor(x,device=currentDevice(),requires_grad=gradP)
        
    return x

def currentDevice():
    
    if(torch.cuda.is_available()):
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def fromTensor(x):
    if(torch.is_tensor(x)):
        return x.data.cpu().numpy()
    return x

def floatTensor(*args):
    dev=currentDevice()
    return(torch.zeros(*args,device=dev,dtype=torch.float32))

def intTensor(*args):
    dev=currentDevice()
    return(torch.zeros(*args,device=dev,dtype=torch.int32))
    
def longTensor(*args):
    dev=currentDevice()
    return(torch.zeros(*args,device=dev,dtype=torch.long))
    
def cudaOnP():
    return (torch.cuda.is_available())

# def makeCudaIfPossible(x):
    
#     if(x is None):
#         return None
#     elif(cudaOnP()):
#         if(isListP(x)):
#             return(map(makeCudaIfPossible,x))
#         if(x.is_cuda):
#             return x
#         else:
#             try:
#                 return x.cuda()
#             except RuntimeError as e: # Out of memory
#                 print('Error in makeCudaIfPossible, probably memory', e)
#                 return None
            
#     else:
#         return x


def mpsOnP(verbP=False):
    
    if torch.backends.mps.is_available():
        return True
    
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+  and/or you do not have an MPS-enabled device on this machine.")
        
    return False

# def isListP(a):
    
#     return(list==type(a))
            