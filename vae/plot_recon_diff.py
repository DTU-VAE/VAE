import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors as c
import matplotlib.patches as mpatches
from scipy import signal

def plot_reconstruction(original, reconstruction, filename="recon_diff", conv_size=3, repeats=3):
	assert isinstance(original, np.ndarray), "'original' argument is not of type 'numpy.ndarray'"
	assert isinstance(reconstruction, np.ndarray), "'reconstruction' argument is not of type 'numpy.ndarray'"
	assert original.shape == reconstruction.shape, "Input arrays have unidentical shape"
	assert conv_size >= 3, "'conv_size' must be greater or equal to 3"
	assert conv_size % 2 == 1, "'conv_size' must be odd number"

	def save_input(src, extension):
		src_plot = src.copy()
		src_plot = src_plot.astype('bool')
		src_plot = np.repeat(src_plot, repeats, axis=1)
		plt.figure(figsize=(10,5))
		plt.axis("off")
		plt.imshow(src_plot, cmap='binary')
		plt.savefig(filename+"_"+extension, dpi=300)


	save_input(original, "original")
	save_input(reconstruction, "reconstruction")

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
	
	# Convolutional operations
	conv_mask = np.ones((conv_size,conv_size), dtype="int8")
	conv_original = signal.convolve2d(original, conv_mask, mode="same", boundary="fill").astype("float")
	original = original.astype('float')
	original[original == 0] = np.NaN
	original[original == 1] = 0
	np.putmask(original, m_in, 1)
	conv_max = np.max(conv_original)
	conv_max = 1 if conv_max == 0 else conv_max
	conv_original -= conv_max
	conv_original /= conv_max
	conv_original[conv_original==0] = 1
	original[m_out] = conv_original[m_out]

	# Tile
	original = np.repeat(original, repeats, axis=1)

	# For print
	plt.figure(figsize=(10,5))
	cMap = c.ListedColormap(['#FF0000','#FF2A00', '#FF5500', '#FF8000', '#FFAA00', '#FFD500',
		'black','chartreuse','chartreuse','chartreuse','chartreuse','chartreuse','chartreuse'])
	patches = [ mpatches.Patch(color="black", label="Original signal"), mpatches.Patch(color="chartreuse", label="Exact reconstruction"), mpatches.Patch(color="#FF0000", label="Inexact reconstruction")]

	plt.axis("off")
	plt.legend(handles=patches, bbox_to_anchor=(.733, 1), loc=2, borderaxespad=0. )	
	plt.imshow(original, cmap=cMap, vmin=-1, vmax=1)

	plt.savefig(filename+"_print", dpi=300)
