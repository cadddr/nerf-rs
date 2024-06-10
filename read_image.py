import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '/System/Volumes/Data/Users/pavlik/Library/Application Support/Kinc Application/default.kha'

def make_image(data, outputname, size=(1, 1), dpi=128):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(outputname, dpi=dpi)
def read_image(infilename=DATA_PATH, width=128, height=128, outfilename='spheres-128-no-shading/image'):
    with open(infilename, 'rb') as file:
        data = file.read()

    bytedata = np.array(bytearray(data))
    num_images = len(bytedata) // (width * height * 4)
    img = bytedata.reshape((num_images, width, height, 4))

    for i in range(num_images):
        make_image(img[i], f'{outfilename}-{i}')
        # fig = plt.figure()
        # plt.imshow(img[i])
        # plt.axis('off')
        # plt.savefig(f'{outfilename}-{i}', bbox_inches='tight')
        # plt.close()
if __name__ == '__main__':
    read_image()
