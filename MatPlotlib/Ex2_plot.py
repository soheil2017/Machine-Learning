import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 5, 11)
y = x ** 2

print(x)
print(y)
print("="*40)

plt.plot(x, y, 'r') # 'r' is the color red
plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show()

#Creating Multiplots on Same Canvas
plt.subplot(1,2,1)
plt.plot(x, y, 'r--') # More on color options later
plt.subplot(1,2,2)
plt.plot(y, x, 'g*-')
plt.show()

# Create Figure (empty canvas)
fig = plt.figure()
# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
# Plot on that set of axes

axes.plot(x, y, 'b')
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')
plt.show(fig)

fig= plt.figure()
axes1= fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes2= fig.add_axes([0.3, 0.3, 0.4, 0.4])
plt.show()

# Creates blank canvas
fig = plt.figure()

axe1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axe2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# Larger Figure Axes 1
axe1.plot(x, y, 'b')
axe1.set_xlabel('X_label_axes2')
axe1.set_ylabel('Y_label_axes2')
axe1.set_title('Axes 2 Title')

# Insert Figure Axes 2
axe2.plot(y, x, 'r')
axe2.set_xlabel('X_label_axes2')
axe2.set_ylabel('Y_label_axes2')
axe2.set_title('Axes 2 Title')
plt.show(fig)
