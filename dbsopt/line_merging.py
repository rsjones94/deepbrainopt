import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ind1 = np.arange(0,100,2)
ind2 = np.arange(-10,110,1)

seq1x = ind1 + np.random.random(len(ind1))*10
seq1y = ind1**2 + np.random.normal(len(ind1))*20

seq2x = ind2 + np.random.random(len(ind2))*2
seq2y = ind2**2.1 - 0.01*ind2**3 + np.random.normal(len(ind2))*20


fig,ax = plt.subplots(1,1)
ax.plot(seq1x,seq1y)
ax.plot(seq2x,seq2y)


fig.show()



'''

Strategies for merging lines

1) Break into points and fit a line?
    The issue is what do you fit. polynomial of nth order?
    Spline?
2) Moving average?


'''