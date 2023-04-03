#!/usr/bin/env python

from rectangle_objects import Rectangle, Box

rectangle = Rectangle(4, 5)
# area = rectangle.getArea()
print(f"The color of rectangle is {rectangle.getColor()}")
print(f"The area of the rectangle is : {rectangle.getArea()}")

# rectangleForStephan = Rectangle(10, 12, "red")
# print(f"The color of rectangle for Stephan is {rectangleForStephan.getColor()}")

box = Box(4, 4, 2)
#area = box.getArea()
#print(f"The area of the box is : {area}")

print(box.color)