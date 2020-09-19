import os
import numpy as np
import matplotlib.pyplot as plt

map05 = [0.926, 0.910, 0.846, 0.761, 0.401]
map05_095 = [0.616, 0.599, 0.502, 0.414, 0.178]
images = [1760, 1232, 880, 538, 176]

p1=plt.plot(images, map05, marker='o', color='r', label='mAP0.5')
p2=plt.plot(images, map05_095, marker='o', color ='b', label='mAP[.5:.05:.95]')
plt.xlabel('training dataset size')
plt.legend(loc='lower right', fancybox=True)

plt.show()
plt.savefig('map.png')