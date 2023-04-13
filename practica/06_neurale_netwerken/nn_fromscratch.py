# De input van ons neural net is een 3x3 rooster waarbinnen de individuele pixels zwart (1) of wit (0) kunnen zijn.
# Hieronder maken we eerst een zwart kruis en een zwart rondje.

########################################################################
import random

########################################################################

cirkel = [
    [1,1,1],
    [1,0,1],
    [1,1,1]
]

kruis = [
    [0,1,0],
    [1,1,1],
    [0,1,0]
]

########################################################################

# In ons netwerk hebben we dus een input layer met 9 inputs (als vector van 9 x 1)
# Om de bovenstaande grids te kunnen lezen als een vector transposen we de input matrix

# Functie om van een 3x3 matrix een 9x1 array te maken
def transpose_vec(input_vec):
    flat_array = []
    for i in input_vec:
        for n in i:
            flat_array.append([n])
    return flat_array

########################################################################

# Elke neuron (of dit nou in eerste of output laag is) heeft 9 weights horend bij elke input neuron. 

# Functie om de weights per neuron random te initialiseren:
def init_weights(size):

    return [[round(random.uniform(0, 1), 2)] for i in range(size)]

# Elke neuron heeft ook een bias die moet worden geïnitialiseerd

def init_bias():

    return round(random.uniform(0, 1), 2)

########################################################################

# We hebben nu dus, als we het houden bij 1 neuron, een input vector:

kruis_plat = transpose_vec(kruis)
print(kruis_plat)

# een weights vector:
weights_voorbeeld = init_weights(len(kruis_plat))
print(weights_voorbeeld)
# en een bias:
bias_voorbeeld = init_bias()
print(bias_voorbeeld)

########################################################################