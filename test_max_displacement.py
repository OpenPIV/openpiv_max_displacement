import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from skimage.registration import phase_cross_correlation
# from skimage import feature
# from skimage.feature.phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage.exposure import match_histograms

from openpiv.tools import imread

# from scipy.fft import rfft2, rfftn

from openpiv.pyprocess import find_subpixel_peak_position
from openpiv.pyprocess import normalize_intensity, fft_correlate_images

a = imread('./test11/A001_1.tif')
# b = imread('../data/PIVChallenge2001_A/A001_2.tif')
winsize = 32
# searchsize = 48
a = a[:winsize,:winsize].copy()

# a[16:18,16:18] = 255
# b = b[:32,:32]

# shift should be in the order of y,x:
# shift = (-12.035, -10.92)

shift = (-0.01, 0.01)

# The shift corresponds to the pixel offset relative to the reference image
b = fourier_shift(np.fft.fftn(a), shift)
b = np.fft.ifftn(b).real
b = match_histograms(b,a).astype('uint8')
# b = b + np.linspace(10,85,32)

fig, ax = plt.subplots(1,2)
ax[0].imshow(a, cmap='gray')
ax[1].imshow(b, cmap='gray')
# plt.colorbar()
plt.show()


a = a[np.newaxis,:,:]
b = b[np.newaxis,:,:]


c1 = fft_correlate_images(a,b,'circular',normalized_correlation=False)
c2 = fft_correlate_images(a,b,'linear',normalized_correlation=False)
c3 = fft_correlate_images(a,b,'circular',normalized_correlation=True)
c4 = fft_correlate_images(a,b,'linear',normalized_correlation=True)

fig,ax = plt.subplots(1,4,figsize=(14,2.5))

counter = 0
for c in [c1,c2,c3,c4]:
    s = ax[counter].contourf(c[0,:,:])
    ax[counter].invert_yaxis()
    plt.colorbar(s, ax=ax[counter])

    default_peak_position = np.floor(np.array(c[0,:,:].shape)/2)
    i = np.array(find_subpixel_peak_position(c[0,:,:]))
    ax[counter].plot(i[1],i[0],'rx')
    print(np.array(i - default_peak_position),
          np.sum(np.abs(np.array(i - default_peak_position)-np.array(shift))))
    counter += 1




image = a[0,:,:]
offset_image = b[0,:,:]
# pixel precision first
shift, error, diffphase = phase_cross_correlation(image, offset_image)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image).conj() * np.fft.fft2(offset_image)
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

plt.show()

print("Detected pixel offset (y, x): {}".format(shift))

# subpixel precision
shift, error, diffphase = phase_cross_correlation(offset_image, image, upsample_factor=100)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
# ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Calculate the upsampled DFT, again to show what the algorithm is doing
# behind the scenes.  Constants correspond to calculated values in routine.
# See source code for details.
# cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
# ax3.imshow(cc_image.real)
# ax3.set_axis_off()
# ax3.set_title("Supersampled XC sub-area")


plt.show()

print("Detected subpixel offset (y, x): {}".format(shift))

