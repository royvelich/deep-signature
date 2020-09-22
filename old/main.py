import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io

# img = io.imread('./turtle.jpg')
# img = rgb2gray(img)
#
# s = np.linspace(0, 2*np.pi, 400)
# r = 300 + 290*np.sin(s)
# c = 300 + 290*np.cos(s)
# init = np.array([r, c]).T
#
# snake = active_contour(gaussian(img, 3),
#                        init, coordinates='rc', max_iterations=20000, convergence=0.01)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
# ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

from skimage import measure


img = io.imread('./hand.jpg')
img = rgb2gray(img)

# Construct some test data
# x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
# r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = measure.find_contours(img, 0.8)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax.plot(contour[0:300, 1], contour[0:300, 0], linewidth=2)
    # ax.plot(contour[900:910, 1], contour[900:910, 0], linewidth=2)
    break

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()