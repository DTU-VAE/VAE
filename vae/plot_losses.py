import numpy as np
import matplotlib.pyplot as plt

train_losses = np.load('../results/losses/train_loss_epoch_1.npy')
valid_losses = np.load('../results/losses/validation_loss_epoch_1.npy')
test_losses  = np.load('../results/losses/test_loss_epoch_1.npy')

plt.figure(figsize=(10,5))
plt.plot(train_losses, 'r--')
plt.plot(valid_losses, 'g-')
plt.plot(test_losses,  'b-')
plt.show()