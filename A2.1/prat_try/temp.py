import numpy as np
import pickle
import matplotlib.pyplot as plt

lambda_g = np.array([4.85, 4.34, 4, 3.49, 3.36])
lambda_a = np.array([3.23, 3.2, 3.07, 2.93, 2.76])
inv_lambda_g2 = 1 / lambda_g**2
inv_lambda_a2 = 1 / lambda_a**2
diff = inv_lambda_a2 - inv_lambda_g2
best_intercept = np.mean(diff)
print(best_intercept)
print('Value of a is ', 0.5/best_intercept ** 0.5)

plt.scatter(inv_lambda_g2, inv_lambda_a2)
x_vals = np.linspace(min(inv_lambda_g2), max(inv_lambda_g2), 100)
y_vals = x_vals + best_intercept
plt.plot(x_vals, y_vals, color='red', label=f'1/λa\u00B2 = 1/λg\u00B2 + 1/(2a)\u00B2 \n---> a = {0.5/best_intercept ** 0.5 :.4f} cm')

# Set the scale of 0.01 units on both axes
plt.gca().set_xticks(np.arange(0.04, 0.16, 0.02))
plt.gca().set_yticks(np.arange(0.04, 0.16, 0.02))

# Set the limits of the axes
plt.gca().set_xlim([0.04, 0.15])
plt.gca().set_ylim([0.04, 0.15])

plt.xlabel('1/λg\u00B2')
plt.ylabel('1/λa\u00B2')
plt.title('Graph showing relation between wavelength in guide and air')
plt.legend()
# plt.grid(True)
plt.show()