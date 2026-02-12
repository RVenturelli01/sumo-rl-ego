from abc import ABC, abstractmethod 


class BaseDone(ABC): 
    
    @abstractmethod 
    def check(self, sim, ego, step_count): 
        pass