#---------------------------------------------------

from Shape import Shape

#---------------------------------------------------

class Button:

    #constructor
    def __init__(self, bgColor, txt, w, h):

        self.backgroundColor = bgColor
        self.text = txt

        self.shape = Shape()
        self.shape.width = w
        self.shape.height = h

    def click(self):

        print(f"Button {self.text}")
        
#---------------------------------------------------

btn_1 = Button(0, "Black", 1, 2)

btn_2 = Button(255, "White")

#---------------------------------------------------

btn_1.click()
btn_2.click()

btn1.shape.width = 100
btn2.shape.height = 100