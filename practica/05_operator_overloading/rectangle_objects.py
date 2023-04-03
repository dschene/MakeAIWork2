# Constructor
# Starts with Double Underscore -> dunder methods
class Rectangle:
    # Indent
    # DocString to document the purpose
    """
    This class can be used to generate Rectangle Objects
    with arguments
    width and height
    """

    # Class variable / Static variable with a value for ALL objects
    color = "black"
    # Default arguments always last
    def __init__(self, widthArg, heightArg):
        """_summary_"""
        # Attributes
        # Encapsulated in object
        self.width = widthArg
        self.height = heightArg
        self.area = round(self.width * self.height)
        #self.color = colorArg

    # Method
    def getArea(self):
        return self.area

    def getColor(self):
        return self.color


# Each class inherits from Rectangle
class Box(Rectangle):
    def __init__(self, widthArg, heightArg, depthArg):
        super().__init__(widthArg, heightArg)
        self.depth = depthArg

    def getDepth(self):
        return self.depth
    


class Triangle:
    def __init__(self, baseArg, heightArg):
        self.base = baseArg
        self.height = heightArg
    
    def get_surface(self):
        return 0.5 * self.base * self.height
    

