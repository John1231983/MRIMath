'''

Class designed to keep track of time and performance. Pretty simple and small, although more capability can be added
if necessary. 
@author Daniel Enrico Cahall

'''


from datetime import datetime

class TimerModule:
    
    start_time = 0.0
    stop_time = 0.0
    
        
    ## Starts the timer
    #
    def startTimer(self):
        self.start_time = datetime.now()
        
    ## Stops the timer
    #
    def stopTimer(self):
        self.stop_time = datetime.now()
    
    ## Computes the amount of time elapsed based on when the timer started and stopped
    # @return the amount of time between the time being started and stopped 
    def getElapsedTime(self):
        return self.stop_time - self.start_time
        
        
        
        