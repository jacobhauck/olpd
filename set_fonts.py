import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # font size of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
