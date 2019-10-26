from matplotlib import pyplot as plt

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(1, figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values, "g", linewidth=2.0)
plt.suptitle('Categorical Plotting')
plt.show()