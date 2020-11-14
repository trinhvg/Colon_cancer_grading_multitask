import json
import numpy as np
import matplotlib.pyplot as plt

json_dir = '/media/vtltrinh/Data1/COLON_PATCHES_1000/log_result/Regres/stats.json'
with open(json_dir) as json_file:
    data = json.load(json_file)
print('conf_matrix:', data['20']['valid-conf_mat'])
a = data['20']['valid-box_plot_data']
data = eval(a)[0]

k = np.asarray(data)
fig1, ax1 = plt.subplots()
ax1.set_title('Regression result')
ax1.set_xticklabels(['BN', 'WD', 'MD', 'PD'])
ax1.yaxis.set_ticks(np.arange(-1, 4, 1))
ax1.yaxis.grid()
ax1.boxplot(k)
plt.show()

fig2, ax2 = plt.subplots()
y = [np.zeros((len(k[i])))+i for i in range(len(data))]
x = np.concatenate(data, axis=0)
y = np.concatenate(y, axis=0)
ax2.set_title('Scatter plot for Four Classes')
ax2.set_xticklabels(['Benign(0)', 'WD(1)', 'MD(2)', 'PD(3)'])
ax2.scatter(y, x)
ax2.yaxis.grid()
plt.axhline(y=0, color='r', linestyle='-')
plt.axhline(y=1, color='r', linestyle='-')
plt.axhline(y=2, color='r', linestyle='-')
plt.axhline(y=3, color='r', linestyle='-')
plt.xticks(ticks=[0, 1, 2, 3])
plt.ylim(-1., 4.)
ax2.yaxis.set_ticks(np.arange(-1, 4, 0.5))
plt.show()