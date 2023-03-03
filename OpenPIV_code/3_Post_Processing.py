# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:40:18 2019

@author: mendez
"""

import os  # This is to understand which separator in the paths (/ or \)

import matplotlib.pyplot as plt  # This is to plot things
import numpy as np  # This is for doing math

################## Post Processing of the PIV Field.

## Step 1: Read all the files and (optional) make a video out of it.

FOLDER = 'C:/US_velocimetry_on_mud/Images_UZL/F4.5_D175/Results_PIV' + os.sep + 'Open_PIV_results_'
n_t = 8  # number of steps.

# Read file number 10 (Check the string construction)
Name = FOLDER + os.sep + 'field_%02d' % 1 + '.txt'  # Check it out: print(Name)
# Read data from a file
DATA = np.genfromtxt(Name)  # Here we have the four colums
nxny = DATA.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
n_s = 2 * nxny
## 1. Reconstruct Mesh from file
X_S = DATA[:, 0]
Y_S = DATA[:, 1]
# Number of n_X/n_Y from forward differences
GRAD_Y = np.diff(Y_S)
# Depending on the reshaping performed, one of the two will start with
# non-zero gradient. The other will have zero gradient only on the change.
IND_X = np.where(GRAD_Y != 0)
DAT = IND_X[0]
n_y = DAT[0] + 1
# Reshaping the grid from the data
n_x = (nxny // (n_y))  # Carefull with integer and float!
Xg = (X_S.reshape((n_x, n_y)))
Yg = (Y_S.reshape((n_x, n_y)))  # This is now the mesh! 60x114.
# Reshape also the velocity components
V_X = DATA[:, 2]  # U component
V_Y = DATA[:, 3]  # V component
# Put both components as fields in the grid
Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
Vxg = (V_X.reshape((n_x, n_y)))
Vyg = (V_Y.reshape((n_x, n_y)))
Magn = (Mod.reshape((n_x, n_y)))

fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
# Plot Contours and quiver
plt.contourf(Xg * 1000, Yg * 1000, Magn)
plt.quiver(X_S * 1000, Y_S * 1000, V_X, V_Y)

###### Step 2: Compute the Mean Flow and the standard deviation.
# The mean flow can be computed by assembling first the DATA matrices D_U and D_V
D_U = np.zeros((n_s, n_t))
D_V = np.zeros((n_s, n_t))
# Loop over all the files: we make a giff and create the Data Matrices
GIFNAME = 'Giff_Velocity.gif'
Fol_Out = 'Gif_Images'
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
images = []

D_U = np.zeros((n_x * n_y, n_t))  # Initialize the Data matrix for U Field.
D_V = np.zeros((n_x * n_y, n_t))  # Initialize the Data matrix for V Field.

for k in range(0, n_t):
    # Read file number 10 (Check the string construction)
    Name = FOLDER + os.sep + 'field_A%03d' % (k + 1) + '.txt'  # Check it out: print(Name)
    # We prepare the new name for the image to export
    NameOUT = Fol_Out + os.sep + 'Im%03d' % (k + 1) + '.png'  # Check it out: print(Name)
    # Read data from a file
    DATA = np.genfromtxt(Name)  # Here we have the four colums
    V_X = DATA[:, 2]  # U component
    V_Y = DATA[:, 3]  # V component
    # Put both components as fields in the grid
    Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
    Vxg = (V_X.reshape((n_x, n_y)))
    Vyg = (V_Y.reshape((n_x, n_y)))
    Magn = (Mod.reshape((n_x, n_y)))
    # Prepare the D_MATRIX
    D_U[:, k] = V_X
    D_V[:, k] = V_Y
    # Open the figure
    fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
    # Or you can plot it as streamlines
    plt.contourf(Xg *1000 , Yg*1000 , Magn)
    # One possibility is to use quiver
    STEPx = 1
    STEPy = 1
    
    plt.quiver(Xg[::STEPx, ::STEPy] * 1000, Yg[::STEPx, ::STEPy] * 1000,
               Vxg[::STEPx, ::STEPy], Vyg[::STEPx, ::STEPy], color='k')  # Create a quiver (arrows) plot
    plt.rc('text', usetex=True)  # This is Miguel's customization
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # ax is an object, which we could get also using ax=plt.gca() 
    # We can now modify all the properties of this obect 
    # In this exercise we follow an object-oriented approach. These
    # are all the properties we modify
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$', fontsize=18)
    ax.set_ylabel('$y[mm]$', fontsize=18)
    #   ax.set_title('Velocity Field via TR-PIV',fontsize=18)
    ax.set_xticks(np.arange(0, 40, 10))
    ax.set_yticks(np.arange(5, 30, 5))
    #   ax.set_xlim([0,43])
    #   ax.set_ylim(0,28)
    #   ax.invert_yaxis() # Invert Axis for plotting purpose
    # Observe that the order at which you run these commands is important!
    # Important: we fix the same c axis for every image (avoid flickering)
    plt.clim(0, 10)
    plt.colorbar()  # We show the colorbar
    plt.savefig(NameOUT, dpi=100)
    plt.close(fig)
    print('Image n ' + str(k) + ' of ' + str(n_t))
    # We append into a Gif

########################################################################
## Now we animate the result #####################################
########################################################################

import imageio  # This used for the animation

GIFNAME = 'Giff_Velocity.gif'
images = []
LENGHT = 10
for k in range(1, LENGHT, 1):
    MEX = 'Preparing Im ' + str(k+1) + ' of ' + str(LENGHT)
    print(MEX)
    FIG_NAME = Fol_Out + os.sep + 'Im%03d' % (k + 1) + '.png'
    images.append(imageio.imread(FIG_NAME))


# Now we can assembly the video and clean the folder of png's (optional)
imageio.mimsave(GIFNAME, images, duration=0.2)
import shutil  # nice and powerfull tool to delete a folder and its content

shutil.rmtree(Fol_Out)

########################################################################
## Compute the mean flow and show it
########################################################################

D_MEAN_U = np.mean(D_U, axis=1)  # Mean of the u's
D_MEAN_V = np.mean(D_V, axis=1)  # Mean of the v's
Mod = np.sqrt(D_MEAN_U ** 2 + D_MEAN_V ** 2)  # Modulus of the mean

Vxg = (D_MEAN_U.reshape((n_x, n_y)))
Vyg = (D_MEAN_V.reshape((n_x, n_y)))
Magn = (Mod.reshape((n_x, n_y)))

fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
# Or you can plot it as streamlines
plt.contourf(Xg * 1000, Yg * 1000, Magn)
# One possibility is to use quiver
STEPx = 1
STEPy = 1
plt.quiver(Xg[::STEPx, ::STEPy] * 1000, Yg[::STEPx, ::STEPy] * 1000,
           Vxg[::STEPx, ::STEPy], Vyg[::STEPx, ::STEPy], color='k')  # Create a quiver (arrows) plot
ax.set_aspect('equal')  # Set equal aspect ratio
ax.set_xlabel('$x[mm]$', fontsize=18)
ax.set_ylabel('$y[mm]$', fontsize=18)
# ax.set_title('Velocity Field via TR-PIV',fontsize=18)
# ax.set_xticks(np.arange(0,40,10))
# ax.set_yticks(np.arange(5,30,5))
#   ax.set_xlim([0,43])
#   ax.set_ylim(0,28)
#   ax.invert_yaxis() # Invert Axis for plotting purpose
# Observe that the order at which you run these commands is important!
# Important: we fix the same c axis for every image (avoid flickering)
plt.clim(0, 10)
plt.colorbar()  # We show the colorbar
plt.savefig('MEAN_FLOW', dpi=100)
plt.close(fig)

## Step 3: Extract three velocity profile. Plot them in self similar forms
# First we select some profiles based on the X=10,20,30 mm
X_LOCs = np.array([10, 20, 30]) / 1000
# Find the corresponding indices: first we take the X axis.
X_axis = Xg[1, :]
# We identify the indices in the mesh where x is the closest to the desired value
Indices = (np.zeros((len(X_LOCs), 1)))  # Initialize the indices vector
# We will store the profiles in a matrix
Prof_U = np.zeros((n_x, len(X_LOCs)))
# The y axis, for plotting purposes is
Y_axis = Yg[:, 1]

for k in range(0, 3):
    Indices[k] = np.abs(X_axis - X_LOCs[k]).argmin()

    # We convert these numbers to make sure they are integer
Indices = Indices.astype(int)

for k in range(0, 3):
    Xdn = Magn[::, Indices[k]]
    Prof_U[:, k] = Xdn[:, 0]

YC = 21.6  # Assume the centerline is approximately at 21.6 mm.
# Obs: for the moment this values is just a quick guess: can you think of a better way to do this?
y_e = Y_axis * 1000 - YC  # This is the experimental grid in mm

fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
plt.plot(y_e, Prof_U[:, 0], 'ko', label='$x_1=10mm$')
plt.plot(y_e, Prof_U[:, 1], 'rs', label='$x_2=20mm$')
plt.plot(y_e, Prof_U[:, 2], 'bv', label='$x_3=30mm$')
ax.set_xlabel('$x[mm]$', fontsize=18)
ax.set_ylabel('$|V|[m/s]$', fontsize=18)
ax.set_title('Velocity Profiles', fontsize=18)
plt.legend()
plt.savefig('Vel_Profiles.png', dpi=100)
plt.show()
plt.close(fig)


# Find the self similarity in these profiles.
# First we fit a Gaussian on these. The function should be
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (sigma ** 2))


# We need this
from scipy.optimize import curve_fit

# We identify the half width parameter from the fitted curves
# These allows for a much finer grid.

Xi = (np.zeros((len(X_LOCs), 1)))  # This is the vector of the half widths

y = np.linspace(-20, 20, 100)  # This will be the fine grid

for k in range(0, 3):
    popt, pcov = curve_fit(gauss, y_e, Prof_U[:, k])  # Do the fitting
    U_FIT = gauss(y, popt[0], popt[1], popt[2])  # Gaussian Fit on profile
    U_FIT = U_FIT / np.max(U_FIT)  # Get the dimensionless velocity
    # It is a simple exercise to show that the half width will be:  
    Xi[k] = np.power(np.log(2) * np.power(popt[2], 2), 0.5)

fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure

plt.plot(y_e / Xi[0], Prof_U[:, 0] / np.max(Prof_U[:, 0]), 'ko', label='$x_1=10mm$')
plt.plot(y_e / Xi[1], Prof_U[:, 1] / np.max(Prof_U[:, 1]), 'rs', label='$x_2=20mm$')
plt.plot(y_e / Xi[2], Prof_U[:, 2] / np.max(Prof_U[:, 2]), 'bv', label='$x_3=30mm$')
ax.set_xlabel('$\hat{x}$', fontsize=18)
ax.set_ylabel('$\hat{U}$', fontsize=18)
ax.set_title('Self Similar Profile', fontsize=18)
plt.legend()
plt.savefig('Self_SIM_Prof.png', dpi=100)
plt.show()
plt.close(fig)
