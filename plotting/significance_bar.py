from matplotlib.markers import TICKDOWN

def plot_bar(ax, start,end,height,displaystring,linewidth = 2, markersize = 8, boxpad =0, fontsize = 14, fontweight = 'bold', color = 'k'):
    "plot significance bars on a given axis"
    bbox_args = dict(facecolor='None', edgecolor='none', boxstyle='Square, pad='+str(boxpad))
    # draw a line with downticks at the ends
    ax.plot([start,end],[height]*2,'-',color = color,lw=linewidth, marker = TICKDOWN, markeredgewidth=linewidth, markersize = markersize)
    # draw the text with a bounding box covering up the line
    ax.text(0.5*(start+end),height+.05,displaystring, ha = 'center',va='center',size = fontsize, weight = fontweight, bbox=bbox_args)