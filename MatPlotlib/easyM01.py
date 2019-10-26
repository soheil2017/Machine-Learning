import pandas as pd
from matplotlib import pyplot as plt

sample_data=pd.read_csv("GT_M01easy.csv")
print(sample_data)

sample_data = sample_data.rename(columns={'field.point.x': 'X'})
sample_data = sample_data.rename(columns={'field.point.y': 'Y'})
sample_data = sample_data.rename(columns={'field.point.z': 'Z'})

plt.plot(sample_data.X, sample_data.Y, "g--", linewidth=2, markersize=1)
#plt.plot(sample_data.X, sample_data.Z, "g--", linewidth=2, markersize=1)

#sample_data.plot.scatter(x='X',y='Y')

plt.legend("easy_MH01")
plt.xlabel("X(meter)")
plt.ylabel("Y(meter)")
plt.title("easy_MH01")
plt.show()

#show=sample_data.head()
#print(show)
#print(sample_data.X.iloc[1])  # shows the second number in X column

#sample_data.ix[0:30].plot.area(alpha=0.4)
#plt.show()

#plt.plot(sample_data['X'])
#plt.show()

