import pandas as pd
from matplotlib import pyplot as plt
x=[1, 2, 3]
y=[1, 4, 9]
z=[10, 5, 0]

plt.plot(x, y, "g--")
plt.plot(x, z, "k--")
#plt.plot(y, z)

plt.title("test plot")
plt.xlabel("X time (second)")
plt.ylabel("Y and Z (meter)")
plt.legend(["Y", "Z"])
plt.show()