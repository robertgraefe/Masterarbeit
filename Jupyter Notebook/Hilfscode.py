# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown

# Function to draw a Piechart
def draw_piechart(arguments):
    fig, ax = plt.subplots(1,len(arguments))
    
    # Handle multiple plots
    try:
        for argument, a in zip(arguments,ax):
            labels = argument[0]
            sizes = argument[1]
            title = argument[2]
            colors = argument[3]
            
            a.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, normalize=False, labeldistance=1.05)
            a.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            a.set_title(title)
            
    # Handle single plot
    except TypeError:
       
        for argument in arguments:
            labels = argument[0]
            sizes = argument[1]
            title = argument[2]
            colors = argument[3]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, normalize=False)
            ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title(title)
    
    plt.show()

    
def cat_compare(count, N, column, title, rest = 0, labels = None):
    COLUMN_NAME = column
    TITLE = title
    REST = rest

    if labels is None:
        LABELS = count.index
        if REST < 0:
            LABELS = LABELS[:REST]
            LABELS = list(LABELS)
            LABELS[-1] = "Rest"
    else:
        LABELS = labels


    SIZES = [count[count.index[idx]] for idx in range(len(count.index))]

    labels = LABELS
    sizes = [element / N for element in SIZES]
    
    if REST < 0:
        stay = sizes[:REST]
        droped = sum(sizes[REST:])
        stay[-1] += droped    
        sizes = stay
        
    return (labels, sizes, title)

# Betrachtung Lorenz-Kurve, Gini, Variationskoeffizient
def lorenz(payback_series, default_series, visual = True):
    
    payback_series = payback_series.dropna()
    default_series = default_series.dropna()
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # Payback
    x = payback_series.sort_values().to_numpy()
    n = len(x)
    v = np.array([x for x in range(n)])
    
    H = v / n
    S = np.cumsum(x)
    Q = S / x.sum()
    VK = x.std()/x.mean()

    Q_ = Q.copy()
    Q_ = np.insert(Q_,0,0)
    Q_ = Q_[:-1]

    P = Q + Q_
    Ph = P / n

    F = 0.5 - 0.5 * Ph.sum()
    G = 2*F
    G_ = n/(n - 1) * G
    
    if visual is True:
        textstr = "Gini = {:.2f} \n VK = {:.2f}".format(G_, VK)

        plt.subplot(1,2,1)
        plt.plot(H,Q)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Payback")

        ax = plt.gca()
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    p = (G_, VK)
    
    # Default
    x = default_series.sort_values().to_numpy()
    n = len(x)
    v = np.array([x for x in range(n)])
    
    H = v / n
    S = np.cumsum(x)
    Q = S / x.sum()
    VK = x.std()/x.mean()

    Q_ = Q.copy()
    Q_ = np.insert(Q_,0,0)
    Q_ = Q_[:-1]

    P = Q + Q_
    Ph = P / n

    F = 0.5 - 0.5 * Ph.sum()
    G = 2*F
    G_ = n/(n - 1) * G
    
    if visual is True:
        plt.subplot(1,2,2)
        plt.plot(H,Q)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title("Default")


        ax = plt.gca()
        ax.plot([0, 1], [0, 1], transform=ax.transAxes)

        textstr = "Gini = {:.2f} \n VK = {:.2f}".format(G_, VK)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    d = (G_, VK)
    
    if visual is True:
        plt.show()
        
    return p, d
