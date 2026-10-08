import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/train.csv')
data = data.sort_values(by = 'Total Costs')
x = np.arange(1, data.shape[0] + 1)

plt.figure(figsize=(10, 6))
plt.scatter(x, data['Total Costs'], s=1, alpha=0.5, color='blue')  # s=1 reduces the size of each point for better performance
plt.title('Scatter Plot of 500,000 Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()
# data.insert(0, 'X', 1)
# x = data.drop(columns='Total Costs').to_numpy()
# y = data['Total Costs'].to_numpy()
# weight = np.loadtxt('data/sample_weights1.txt')

# xn = x.T.copy()
# xn = (xn)*weight
# w = (np.linalg.inv(xn@x)@xn)@y

# data_new = pd.read_csv('data/test.csv')
# data_new.insert(0, 'X', 1)
# x_new = data_new.to_numpy()
# # print(x_new.shape)
# np.savetxt('out.txt', x_new@w)