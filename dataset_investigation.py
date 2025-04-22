"""
В данном скрипте производиться первичный анализ исследуемых данных (laptop_price.csv).
Что соответствует второму и третьему пунктам домашнего задания.
В пункте втором меются следующие подпункты:
 - Вывести информацию о столбцах.
 - Построить гистограмму распределения стоимости ноутбуков.
 - Нарисовать тепловую карту корреляции.
 - Указать инсайты (результатов анализа), полученные из текущего анализа.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

work_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(work_dir, 'laptop_price.csv')
df = pd.read_csv(path, encoding='latin-1')
label_column = 'Price_euros'

print(df.info())
print(df.head())

print(f'Количество строк в dataset: {df.shape[0]}')
for column in df.columns:
    print(f'Количество уникальных значений в колонке {column}: {df[column].nunique()}')

input("Press Enter to continue...")

"""
Исходя из колонок, которые содержит dataset, очевидно, что колонки 'Company', 'TypeName', 'OpSys', 'Gpu', 'Cpu', 'Ram' и ScreenResolution требуют замены на числовые значения.
Также колонка 'laptop_ID' не несет никакой информации, поэтому ее можно удалить.
Каждое второе значения в колонке 'Product' является уникальным, по всей видимости, это не имеет явного влияния на цену, поэтому ее также можно удалить.
Значения в колонке 'Weight' разумно преобразовать в числовые, так как это "снизит" уникальность значений.
Стоит отметить, что для колонки 'Memmory' такой подход не целесообразен ввиду умеренной уникальности значений и наличия дополнительных сведений (кроме объема ПЗУ).
"""

df.drop(columns=['laptop_ID', 'Product'], inplace=True)
named = ['Company', 'TypeName', 'OpSys', 'Gpu', 'Cpu', 'Ram', 'Memory', 'ScreenResolution']

le = LabelEncoder()
for column in named:
    df[column] = le.fit_transform(df[column])

df['Weight'] = df['Weight'].apply(lambda weight: float(weight.replace('kg', '').replace(' ', '')))

# Рассмотрим распределения численных значений в dataset.
print(df.drop(columns=named).describe())

# Теперь можно рассмотреть степень корреляции между значениями колонок.
sns.heatmap(df.corr(numeric_only=True), cmap="Blues", annot=True)
plt.savefig(os.path.join(work_dir, 'coverage_map.png'), dpi=1200)
plt.show()

"""
Примем в качестве высокой степени корреляции для значений коэффициента ковариации выше 0.75.
В первую очередь, рассмотрим степень корреляции между стоимостью и другими значениями.
Как можно заметить, ни от одного параметра не наблюдается высокая степень ковариации.
В таком случае, рассмотрим влияние, которое вносят наибольший вклад в цену. Это значения CPU (0.53) и GPU (0.44).

Теперь найдем колонки, для которыйх коэффициент ковариации имеет высокое значение (0.75).
Такими колонками являются только CPU и GPU.
Теперь построим графики стоимости в завистмости от значений колнок CPU и GPU. А так же построим график зависимости между CPU и GPU.
Данные графики помогут выбрать алгоритм аппроксимации между исходными параметрами и ценой. 
"""

plt.plot(df['Cpu'], df[label_column], 'o', label='Cpu')
plt.xlabel('Cpu')
plt.ylabel('Price')
plt.show()

plt.plot(df['Gpu'], df[label_column], 'o', label='Gpu')
plt.xlabel('Gpu')
plt.ylabel('Price')
plt.show()

plt.plot(df['Gpu'], df['Cpu'], 'o')
plt.xlabel('Gpu')
plt.ylabel('Cpu')
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(df['Cpu'], df['Gpu'], df[label_column])
ax.set_xlabel('Cpu')
ax.set_ylabel('Gpu')
ax.set_zlabel('Price')

ax.view_init(0, 0)
plt.show()

"""
К сожалению, графики не дают четкого представления о зависимости между параметрами.
"""

sns.histplot(data = df[label_column], kde = True)
plt.show()

"""
Гистограмма чила ноутбуков в зависимости от цены (ее промежутка) показывает, что большинство ноутбуков имеют цену в диапазоне от 0 до 3000 евро.
Построим график значений ковариаций между стоимостью и остальными колонками.
"""

cut_df = df[df[label_column] < 3000]
covs = cut_df.corr(numeric_only=True)[label_column].drop(label_column).sort_values(ascending=True)
plt.plot(covs, 'o', label='covs')
plt.xlabel('Columns')
plt.ylabel('Covariance')
plt.grid()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(df['Cpu'], df[label_column], 'or')
axes[0].set_xlabel('Cpu')
axes[0].set_ylabel('Price')

axes[0].plot(cut_df['Cpu'], cut_df[label_column], '+b')
axes[0].set_xlabel('Cpu')
axes[0].set_ylabel('Price')

axes[1].plot(df['Gpu'], df[label_column], 'or')
axes[1].set_xlabel('Gpu')
axes[1].set_ylabel('Price')

axes[1].plot(cut_df['Gpu'], cut_df[label_column], '+b')
axes[1].set_xlabel('Gpu')
axes[1].set_ylabel('Price')
plt.show()

"""
Увы, даже после того, как были срезаны значения, которые превышают 3000 евро, графики не дают четкого представления о характере зависимости между GPU/CPU и стоимостью ноутбука.
Как следствие, сложно сделать выбор между доступными нам моделями: LR, DTR и KNN.
Далее попробуем нормировать все значения и сократить колличество параметров применением метода главных компонент.
Выше указанные действия соответствуют пункту два домашенего задания.
В третьем пункте содержаться следующие подпункты:
 - Выполнить масштабирование признаков (StandardScaler).
 - Применить PCA так, чтобы сохранить ≥95% дисперсии.
 - Сколько компонент сохранилось? Какие?
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .model_trainer import ModelTrainer
from sklearn.linear_model import LinearRegression, DecisionTreeRegressor, KNeighborsRegressor

pca = PCA(n_components=0.95, random_state=1)
ss = StandardScaler()
X = pca.fit_transform(ss.fit_transform(df.drop(columns=[label_column])))
y = df[label_column]

plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Количество компонент")
plt.ylabel("Накопленная доля объяснённой дисперсии")
plt.title("PCA")
plt.grid(True)
plt.show()

"""
Согласно полученному графику PCA, для сохранения 95% дисперсии необходимо использовать 8 компонент. Однако, если снизить требование до 93%, то можно обойтись 7 компонентами.
Полагаю, что для обучения наших моделей 93% дисперсии будет достаточно.
"""

pca = PCA(n_components=0.93, random_state=1)
X = pca.fit_transform(ss.fit_transform(df.drop(columns=[label_column])))
# Тут необходимо добавить список оставленных комполнент.

"""
Поскольку в рамках домашнего задания нужно сделать выбор между моделями обучения, а исходя из предыдущео анализа я не могу определить оптимальную модель.
Для выбора модели обучим все три и сравним результаты предсказания с точки зрения точности предсказания и степени переобученности/недообученности модели.
Для оценки точности предсказания будем использовать R2 и RMSE:
 - R2, так как этот показатель дает понять, насколько адекватно отданные параметры объясняют дисперсию предсказанного значения.
 - RMSE, так как этот показатель дает понять, насколько предсказанное значение отличается от реального значения.

Для оценки степени переобученности вектор меток и исходных значений (Преобразованный dataset разобъем на три игтервала: обучающая, тестовая и контрольная выборки).
А поле завершения обучения модели, проверим ее точность на контрольной выборке.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
prepared_X={'train': X_train, 'test': X_test, 'valid': X_valid}, prepared_y={'train': y_train, 'test': y_test, 'valid': y_valid}

trainers = {
    'LinearRegression': ModelTrainer(prepared_X, prepared_y, LinearRegression()),
    'DecisionTreeRegressor': ModelTrainer(prepared_X, prepared_y, DecisionTreeRegressor()),
    'KNeighborsRegressor': ModelTrainer(prepared_X, prepared_y, KNeighborsRegressor())
}

# Сохраним теперь скорректированные значения в отдельный файл.
df.to_csv(os.path.join(work_dir, 'laptop_price_cleaned.csv'), index=False)
