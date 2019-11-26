import numpy as np
import matplotlib.pyplot as plt

losses = np.load('../results/losses/loss_epoch_1.npy')

plt.figure()
plt.plot(losses, 'r')
plt.show()