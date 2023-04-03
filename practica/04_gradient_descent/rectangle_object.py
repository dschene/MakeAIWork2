class Rectangle:

    def __init__(self, widthArg, heightArg, coloArg = 'black'):
        #attributes
        self.width = widthArg
        self.height = heightArg
        self.area = round(self.width * self.height)
        self.color = coloArg

    
    def getArea(self):
        return self.area
    
    def getColor(self):
        return self.color
    

class Box(Rectangle):

    def __init__(self, widthArg, heightArg, depthArg):
        super().__init__(widthArg, heightArg)
        self.depth = depthArg


