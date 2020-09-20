import os
import matplotlib.pyplot as plt
plt.style.use('myfig.mplstyle')

def save_figure(fig, name):
    IMG_FOLDER = r'../py_imgs'
    imgname = name + '.png'
    fig.savefig(os.path.join(IMG_FOLDER, imgname))
