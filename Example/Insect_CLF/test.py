import numpy as np

a = [[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]]
b = [[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]]
c = [[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]]
d = [[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]]
a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)

m = np.expand_dims(a,axis=0)
# m = np.append(b,axis=0)

# print(m)

a = [1,2,3]
b = [5,6,7]

for i,j in zip(a,b):
    print(i,j)