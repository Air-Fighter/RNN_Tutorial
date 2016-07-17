# coding=utf-8
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import font_manager

chf = font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')

def plotfig2file(data, file_name='tmp.png', x_labels=None, y_labels=None, color_mode='RdBu_r', vmin=None, vmax=None,
                 color_degree=100, ):
    y_inch = len(y_labels) / 3

    fig = plt.figure(facecolor='w', figsize=(8, y_inch), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)

    if vmin is None:
        vmin = data[data != 0].min()
    if vmax is None:
        vmax = data[data != 0].max()

    if not x_labels is None:
        if not len(x_labels) == data.shape[1] and not x_labels[0] == '':
            print "dimension error: please check data.shape and len(x_labels)"
            exit(-1)
        if not x_labels[0] == '':
            x_labels.insert(0, '')
        ax1.set_xticklabels(x_labels[1:], range(len(x_labels) - 1), fontproperties=chf)
    if not y_labels is None:
        if not len(y_labels) == data.shape[0] and not y_labels[0] == '':
            print "dimension error: please check data.shape and len(y_labels)"
            exit(-1)
        if not y_labels[0] == '':
            y_labels.insert(0, '')
        ax1.set_yticks(xrange(0,len(y_labels),1))
        ax1.set_yticklabels(y_labels[1:], fontproperties=chf)

    cmap = cm.get_cmap(color_mode, color_degree)

    map = ax1.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=1)
    # cb.set_label('(%)')

    plt.savefig(file_name, format='png')
    # plt.show()

if __name__ == '__main__':
    A = np.random.standard_normal((25, 6))
    x_labels = [u'世纪', u'B', u'C', u'D', u'E', u'F']
    fig_name = './test.png'
    plotfig2file(data=A, file_name=fig_name, x_labels=x_labels)