import numpy as np
import csv
import matplotlib.pyplot as plt

age_list = np.array([])
salary_list = np.array([])

with open('Football_players.csv', encoding='latin-1') as f:
    csv_list = list(csv.reader(f))

for row in csv_list[1:]:
    age_list = np.append(age_list, int(row[4]))
    salary_list = np.append(salary_list, int(row[8]))



def simlin_coef(x, y):

    x_mean = np.average(x)
    y_mean = np.average(y)

    a = 0
    b = 0
    for i in range(len(x)):
        a = a + (x[i] - x_mean)*(y[i] - y_mean)
        b = b + (x[i] - x_mean)**2

    b1 = a / b
    b0 = y_mean - b1*x_mean

    return b1, b0

b1, b0 = simlin_coef(age_list, salary_list)

def simlin_plot(x, y, b1, b0):
    plt.figure()
    plt.scatter(x, y, c='b')
    ploty = b1*x + b0
    plt.plot(x, ploty, color='r')
    plt.ylabel("Salary")
    plt.xlabel("Age")
    plt.title("Simple Linear Regression: Age vs Salary")
    plt.show()

simlin_plot(age_list, salary_list, b1, b0)