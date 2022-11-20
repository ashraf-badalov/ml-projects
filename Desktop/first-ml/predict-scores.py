import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20,50,15,67,39,28,48,56,10,20,30,33,17]).reshape(-1,1)
scores = np.array([56,96,45,100,75,67,90,96,35,45,59,62,40]).reshape(-1,1)

                     # Visualizing the model and predict
# model = LinearRegression()
# model.fit(time_studied,scores)

# plt.scatter(time_studied,scores)
# plt.plot(np.linspace(0,70,100).reshape(-1,1),model.predict(np.linspace(0,70,100).reshape(-1,1)),'r')
# plt.ylim(0,100)
# plt.show()

# print(model.predict(np.array([25]).reshape(-1,1)))

                                  # Testing

time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.3)

model = LinearRegression()
model.fit(time_test, score_test)
print(model.score(time_test, score_test))

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.show()