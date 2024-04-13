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

print("Showing original results:")
coef_list = calculate_coeff(matrix, salary_list)
y = calculate_predictions(matrix, coef_list)
print(calculate_mean_square_error(y, salary_list))

print("Showing results with an added random column:")
new_matrix = add_random_column_to_matrix(matrix)
new_coeflist = calculate_coeff(new_matrix, salary_list)
y = calculate_predictions(new_matrix, new_coeflist)
print(calculate_mean_square_error(y, salary_list))

print("Showing results with an added random column:")
new_matrix = add_random_column_to_matrix(matrix)
new_coeflist = calculate_coeff(new_matrix, salary_list)
y = calculate_predictions(new_matrix, new_coeflist)
print(calculate_mean_square_error(y, salary_list))




