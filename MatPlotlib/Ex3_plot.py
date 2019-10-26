import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 5, 11)
y = x ** 2

# Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0][0].plot(x,y, 'r')
axes[0][1].plot(x,y, 'r-')
axes[1][0].plot(x,y)
axes[1][1].plot(x,y)
plt.show()


fig1, axes = plt.subplots(nrows=3, figsize=(8,2))

axes[0].plot(x, y, 'r')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('title')

axes[1].plot(x, y, 'r')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

axes[2].plot(x, y, 'r')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')

plt.show()

fig2, ax = plt.subplots(figsize=(12,6))

ax.plot(x, x+1, color="red", linewidth=0.25)
ax.plot(x, x+2, color="red", linewidth=0.50)
ax.plot(x, x+3, color="red", linewidth=1.00)
ax.plot(x, x+4, color="red", linewidth=2.00)

# possible linestype options ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’
ax.plot(x, x+5, color="green", lw=3, linestyle='-')
ax.plot(x, x+6, color="green", lw=3, ls='-.')
ax.plot(x, x+7, color="green", lw=3, ls=':')

# custom dash
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...
ax.plot(x, x+ 9, color="blue", lw=3, ls='-', marker='+')
ax.plot(x, x+10, color="blue", lw=3, ls='--', marker='o')
ax.plot(x, x+11, color="blue", lw=3, ls='-', marker='s')
ax.plot(x, x+12, color="blue", lw=3, ls='--', marker='1')

# marker size and color
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8,
        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="green");
plt.show(fig2)


