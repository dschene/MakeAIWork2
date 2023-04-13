dt = 0.01
t  = 0

#Define the car
F_motor = 800
mass = 600
a = F_motor / mass

v = 0
x = 0
running = True

while running:

    t = t + dt

    dv = a * dt
    v += dv

    dx = v * dt
    x = x + dx
    
    

    if x >= 100:
        print(t)
        break



