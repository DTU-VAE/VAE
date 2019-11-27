import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors as c


def plot_reconstruction(original, reconstruction, save=False, filename="recon_diff"):
	assert isinstance(original, np.ndarray), "'original' argument is not of type 'numpy.ndarray'"
	assert isinstance(reconstruction, np.ndarray), "'reconstruction' argument is not of type 'numpy.ndarray'"
	assert original.shape == reconstruction.shape, "Input arrays have unidentical shape"

	# Mask function
	def mask_op_source(src, mask, op):
		assert op in ["eq", "out"], "Undefined mask operation"

		src = src.astype('bool')
		mask = mask.astype('bool')

		if op == "eq":
			return np.logical_and(src, mask)
		if op == "out":
			return np.logical_and(mask, ~src)

	# Create image
	original = original.astype('uint8')
	reconstruction = reconstruction.astype('uint8')
	m_in = mask_op_source(original, reconstruction, "eq")
	m_out = mask_op_source(original, reconstruction, "out")
	np.putmask(original, m_in, 2)
	np.putmask(original, m_out, 3)

	# Plot image
	cMap = c.ListedColormap(['black','white','green', 'red'])
	plt.figure(figsize=(16,8))
	plt.imshow(original, cmap=cMap, vmax=4)
	plt.axis("off")
	if save:
		plt.savefig(filename, dpi=300)
	plt.show()
