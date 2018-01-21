'''
Created on Jan 12, 2018

@author: daniel
'''
from tensorflow.python.client import device_lib
import multiprocessing
from multiprocessing.Pool import Pool as ThreadPool 


class HardwareHandler:
    
    pool = None
    
    def __init__(self):
        self.numThreads = self.getNumberOfCores()
    
    def getAvailableGPUs(self):
        local_device_protos = device_lib.list_local_devices()
        return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    
    def getNumberOfCores(self):
        return multiprocessing.cpu_count()

    def createThreadPool(self, threadCount = None):
        if threadCount is None:
            threadCount = self.getNumberOfCores()
        if self.pool == None:
            self.pool = ThreadPool(threadCount) 
        return self.pool

        
        
