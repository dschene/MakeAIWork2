# class Rectangle:

#     def __init__(self, width, height):
#         self.width = width
#         self.height = height

class Pet_class:
    """This class can be used to create pet objects"""
    
    def __init__(self, naam):
        self.name = naam

    def make_noise(self, n_of_times = 5, upper_case = False):
        '''Barks a given amount of items'''
        
        if upper_case == True:
        
            return f"{self.name} {('woef' * n_of_times).upper()}"
        
        else:
            return f"{self.name} {'woef' * n_of_times}"