'''
Created on Jan 6, 2018

@author: daniel
'''

from datetime import datetime

class TimerModule:
    
    start_time = 0.0
    stop_time = 0.0
    
    
    def startTimer(self):
        self.start_time = datetime.now()
        
        
    def stopTimer(self):
        self.stop_time = datetime.now()
        
    def getElapsedTime(self):
        return self.stop_time - self.start_time
        
        
        
        