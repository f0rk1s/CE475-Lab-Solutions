import numpy as np
import csv
import matplotlib.pyplot as plt

age_list = np.array([])
height_list = np.array([])
mental_list = np.array([])
skill_list = np.array([])
salary_list = np.array([])
one_list = np.array([])

with open('Football_players.csv', encoding='latin-1') as f:
    csv_list = list(csv.reader(f))


for row in csv_list[1:]:
    age_list = np.append(age_list, int(row[4]))
    height_list = np.append(height_list, int(row[5]))
    mental_list = np.append(mental_list, int(row[6]))
    skill_list = np.append(skill_list, int(row[7]))
    salary_list = np.append(salary_list, int(row[8]))
    one_list = np.append(one_list, 1)

#matrix
matrix = np.column_stack((one_list, age_list, height_list, mental_list, skill_list))

def calculate_coeff(x, salary_list):
    coef_list = np.array([])
    coef_list = x.transpose()
    coef_list = coef_list.dot(x)
    coef_list = np.linalg.inv(coef_list)
    coef_list = coef_list.dot(x.transpose())
    coef_list = coef_list.dot(salary_list)
    return coef_list

def calculate_mean_square_error(y_hat, y_true):
    return np.mean((y_hat - y_true) ** 2)

def add_random_column_to_matrix(matrix):
    new_column = np.array([])
    for row in matrix:
        new_column = np.append(new_column, np.random.randint(-1000, 1000))
    new_matrix = np.column_stack((matrix, new_column))
    return new_matrix

def calculate_predictions(matrix, coef_list):
    predictions = np.array([])
    for row in matrix:
        predictions = np.append(predictions, np.dot(row, coef_list))

    return predictions

def remove_Row_And_anElement(matrix, y, k):
    #remove kth row from the matrix
    new_matrix = np.delete(matrix, k, 0)
    new_vector = np.delete(y, k, 0)
    return matrix[k], y[k], new_matrix, new_vector

def cross_validation(matrix, y):
    mse_loocv = np.array([]) #array to store 100 mse.
    for n in range (0, len(matrix)):
        test_input, test_output, train_input, train_output = remove_Row_And_anElement(matrix, y, n)
        coef_list = calculate_coeff(train_input, train_output)
        predictions = np.dot(test_input, coef_list)
        mse_loocv = np.append(mse_loocv, calculate_mean_square_error(predictions, test_output))

    mse_avg = np.mean(mse_loocv)
    return mse_loocv, mse_avg


mse_loocv, mse_avg = cross_validation(matrix, salary_list)

print("Mean squared error using Leave-One-out cross-validation:", mse_avg)


coef_list = calculate_coeff(matrix, salary_list)
y = calculate_predictions(matrix, coef_list)
mse_all = calculate_mean_square_error(y, salary_list)
print("Mean squared error using training error only", mse_all)


plt.figure()
oneToHundred = np.linspace(1,100,100)
plt.scatter(oneToHundred, mse_loocv, c='r', label = "MSE for each test data")
plt.axhline(y=0, c='b', label = "Average MSE with LOOCV")
plt.axhline(y=mse_avg, c='g', label = "Training MSE")
plt.axhline(y=mse_all, c='k', label = "Zero error line")
plt.ylabel("MSE (per 10 million)")
plt.xlabel("Data point no.")
plt.title("LOOCV: MSE Comparison")
plt.legend()
plt.show()
