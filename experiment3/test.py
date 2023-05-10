import matplotlib.pyplot as plt
import numpy as np


loss_dict = {"0.1": 20, "0.01": 60, "0.001": 10}

myList = loss_dict.items()

x, y = zip(*myList)
ASDAS = 3
plt.plot(x, y)
plt.xlabel('Key')
plt.ylabel('Value')
plt.yticks(np.arange(0, 105, 5))
plt.title(F'My Dictionary: {ASDAS}')
plt.show()

