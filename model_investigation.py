"""
В данном скрипте будет производиться исследование выбранной в результате анализа данных модели.
В данном случае, речь идет о KNN модели.
Будет произведена оценка точности обучения в зависимости от гиперпараметров, передаваемых модели.
И сравнение наилучшего результата с результатом полученном на этапе выбора модели.
Отметим, что на прошлом этапе для учебной выборки RMSE = 433 и R² = 0.65
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X = np.loadtxt('X.csv', delimiter=',')
y = np.loadtxt('y.csv', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

param_grid = { 'n_neighbors': range(3,11,2), 'weights': ['uniform', 'distance'], 'p': [1, 2] }
grid_search = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(X_train, y_train)

print(f"Best R2 score: {grid_search.best_score_}")
print(f"Best hyperparameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"R2 score on test set: {r2_score(y_test, y_pred)}")

"""
Как можно заметить, код не отличается по существу от кода, который был приведен в лекции.
Это ожидаемо, так как пространство для маневра невелико.
Тем не менее, поиск гиперпараметров дал положительные результаты, увеличив показатель R² до 0.75.
В качестве вывода можно разве что отметить, что подбор гиперпараметров был не напрасен. Другие выводы мне сделать сложно.
Возможно, ввиду недостатка опыта. Или стоит проверить иные модели.
"""

"""
Пункт 6. Общие выводы.

"""