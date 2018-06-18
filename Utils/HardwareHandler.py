'''

Class designed to handle all hardware related tasks, such as creating a threadpool or getting the number of cores or GPUs.
I anticipate that this class will grow over time, but for the time being it handles all necessary hardware tasks.

@author Daniel Enrico Cahall

'''
from tensorflow.python.client import device_lib
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 


class HardwareHandler:
    
    pool = None
    
    ## The constructor for the hardwarehandler class. Sets the default number of threads to the number of available cores
    def __init__(self):
        self.numThreads = self.getNumberOfCores()
        
    
    ## Acquires the number of GPUs available to use
    #
    # @return the available number of GPUs on the device (int)
    def getAvailableGPUs(self):
        local_device_protos = device_lib.list_local_devices()
        return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    
        
    ## Acquires the number of CPU cores to use
    #
    # @return the number of cores on the device (int)
    def getNumberOfCores(self):
        return multiprocessing.cpu_count()

    
    ## Creates a threadpool, where the number of threads is the number of available cores by default
    #
    # @param threadCount the number of threads to use (default is number of cores)
    # @return pool the threadpool which was created
    def createThreadPool(self, threadCount = None):
        if threadCount is None:
            threadCount = self.getNumberOfCores()
        if self.pool == None:
            self.pool = ThreadPool(threadCount) 
        return self.pool

        
        
