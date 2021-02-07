import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 

filename = "./mlm.csv"
data = pd.read_csv(filename)
x1_ary = np.array(data.iloc[0:800, 0].values)
x2_ary = np.array(data.iloc[0:800, 1].values)
y_ary = np.array(data.iloc[0:800, 2].values)
x1 = x1_ary.reshape(800, 1)
x2 = x2_ary.reshape(800, 1)
x_ary = np.hstack((x1, x2, np.ones((800, 1), dtype = float)))
x_test = data.iloc[800: 1001, 0]
y_test = data.iloc[800: 1001, 1]
z_test = data.iloc[800: 1001, 2]

# 正规方程法 matirx method
X = np.matrix(x_ary)
Y = np.matrix(y_ary.reshape(800, 1))
W = X.T.dot(X).I.dot(X.T).dot(Y)
e = np.sum(pow(np.array(X.dot(W) - Y), 2))/len(x1)/2
w = np.array(W)
print(w)
print("MSE:", e)
ax = plt.axes(projection="3d")
ax.scatter3D(x_test, y_test, z_test)
draw_x = np.linspace(0, 90)
draw_y = np.linspace(0, 90)
X_drawing, Y_drawing = np.meshgrid(draw_x, draw_y)
ax.plot_surface(X=X_drawing,Y=Y_drawing,Z=X_drawing * w[0] + Y_drawing * w[1] + w[2],color='b',alpha=0.2)
ax.view_init(elev=30, azim=30)
plt.show()

# neural networks method
'''
Linear = linear_model.LinearRegression()
Linear.fit(x_ary, y_ary)
ax = plt.axes(projection="3d")
ax.scatter3D(x_test, y_test, z_test)
draw_x = np.linspace(0, 90)
draw_y = np.linspace(0, 90)
X_drawing, Y_drawing = np.meshgrid(draw_x, draw_y)
ax.plot_surface(X=X_drawing,Y=Y_drawing,Z=X_drawing * Linear.coef_[0] + Y_drawing * Linear.coef_[1] + Linear.intercept_,color='r',alpha=0.2)

ax.view_init(elev=30, azim=30)
plt.show()
'''