import matplotlib.pyplot as plot
import numpy as np

for i in range(500,4000,500):
    q = np.load(f'data/{i:05}.npy')
    print(np.min(q))