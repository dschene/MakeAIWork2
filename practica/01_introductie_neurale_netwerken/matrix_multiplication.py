m_1 =[[1,2,3],
      [4,5,6]]

m_2 = [[10,11],
       [20,21],
       [30,31]]

def m_multi(m1, m2):
    output_matrix = []
    
    if len(m1[0]) == len(m2):

        for i, n in enumerate(m_1):
            l = []
            for x in range(len(m2[0])):
                to_add = 0
                for z, y in enumerate(m_2):
                    to_add += (n[z] * y[x])

                l.append(to_add)

            output_matrix.append(l)       
        
        return output_matrix
    
    else:
        return 'Matrices not multipliable'

                
print(m_multi(m_1, m_2))