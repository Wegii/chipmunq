
import os
import math

from itertools import product
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.markers as mmark
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import numpy as np


d_list = [3, 5, 7]
noiss_list = [0.001+ 0.0005 * i for i in range(12)]
framework_list = ['CodeStitch', 'SurfStitch']

data = { 
    ('CodeStitch', 3) : [],
    ('CodeStitch', 5) : [],
    ('CodeStitch', 7) : [],
    ('SurfStitch', 3): [0.1+ 0.0005 * i for i in range(12)],
    ('SurfStitch', 5): [0.01+ 0.0005 * i for i in range(12)],
    ('SurfStitch', 7): [0.001+ 0.0005 * i for i in range(12)],
}


marker_list = ['o', '^']
linestyle_list = ['solid', 'dotted']
color_list = ['C0', 'C1', 'C2']


fig, ax1 = plt.subplots(1, 1, frameon=True)


frame_lw = 1.5
labelpad = 7

markersize = 7
line_width = 2.2

label_font = 14
param_font = 12
legend_font = 12

for framework, d in product(framework_list, d_list):
    ax1.plot(noiss_list, data[(framework, d)], 
            marker=marker_list[framework_list.index(framework)], markersize=markersize,
            linestyle=linestyle_list[framework_list.index(framework)], linewidth=line_width,
            color=color_list[d_list.index(d)])
    
ax1.set_yscale('log')

ax1.tick_params(width=frame_lw, labelsize=param_font)
plt.setp(ax1.spines.values(), lw=frame_lw)
ax1.grid(color='lightgrey', axis='y', linestyle='solid', linewidth=frame_lw)

ax1.set_xlabel('Count of defective qubits', fontsize=label_font, labelpad=labelpad)
ax1.set_ylabel('Logical error rate per circle', fontsize=label_font, labelpad=labelpad)

custom_lines = [
    Line2D(
        [0], [0], 
        marker=marker_list[framework_list.index(framework)], markersize=markersize,
        linestyle=linestyle_list[framework_list.index(framework)], linewidth=line_width,
        color='black',
        label='Classical LS' if framework == 0 else 'Surf-Deformer'
    ) 
    for framework in framework_list
]
custom_legend = custom_lines + [mpatches.Patch(color=color_list[d_list.index(d)], label='d = %d code' % d) for d in d_list] 

custom_legend = [custom_legend[i] for i in [0, 2, 4, 1, 3, 5]]

ax1.legend(handles=custom_legend, 
          ncol=2, fontsize=10,
          edgecolor='darkgray',
          facecolor='white', framealpha=0.8).get_frame().set_linewidth(frame_lw)

fig.set_dpi(400)
fig.savefig(os.path.join(os.getcwd(), "Data", "a" +".pdf"), transparent=True, bbox_inches='tight',format='pdf')