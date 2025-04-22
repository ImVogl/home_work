"""
В данном скрипте производиться первичный анализ исследуемых данных (laptop_price.csv).
Что соответствует первому пункту домашнего задания.
Имеются следующие подпункты:
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
named = ['Company', 'TypeName', 'OpSys', 'Gpu', 'Cpu', 'Ram', 'Memory']

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

plt.plot(df['Cpu'], df['Price_euros'], 'o', label='Cpu')
plt.xlabel('Cpu')
plt.ylabel('Price')
plt.show()

plt.plot(df['Gpu'], df['Price_euros'], 'o', label='Gpu')
plt.xlabel('Gpu')
plt.ylabel('Price')
plt.show()

plt.plot(df['Gpu'], df['Cpu'], 'o')
plt.xlabel('Gpu')
plt.ylabel('Cpu')
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(df['Cpu'], df['Gpu'], df['Price_euros'])
ax.set_xlabel('Cpu')
ax.set_ylabel('Gpu')
ax.set_zlabel('Price')

ax.view_init(0, 0)
plt.show()

"""
К сожалению, графики не дают четкого представления о зависимости между параметрами.
"""

sns.histplot(data = df['Price_euros'], kde = True)
plt.show()

"""
Гистограмма чила ноутбуков в зависимости от цены (ее промежутка) показывает, что большинство ноутбуков имеют цену в диапазоне от 0 до 3000 евро.
Построим график значений ковариаций между стоимостью и остальными колонками.
"""

cut_df = df[df['Price_euros'] < 3000]
covs = cut_df.corr(numeric_only=True)['Price_euros'].drop('Price_euros').sort_values(ascending=True)
plt.plot(covs, 'o', label='covs')
plt.xlabel('Columns')
plt.ylabel('Covariance')
plt.grid()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(df['Cpu'], df['Price_euros'], 'or')
axes[0].set_xlabel('Cpu')
axes[0].set_ylabel('Price')

axes[0].plot(cut_df['Cpu'], cut_df['Price_euros'], '+b')
axes[0].set_xlabel('Cpu')
axes[0].set_ylabel('Price')

axes[1].plot(df['Gpu'], df['Price_euros'], 'or')
axes[1].set_xlabel('Gpu')
axes[1].set_ylabel('Price')

axes[1].plot(cut_df['Gpu'], cut_df['Price_euros'], '+b')
axes[1].set_xlabel('Gpu')
axes[1].set_ylabel('Price')
plt.show()

"""
Увы, даже после того, как были срезаны значения, которые превышают 3000 евро, графики не дают четкого представления о характере зависимости между GPU/CPU и стоимостью ноутбука.
Как следствие, сложно сделать выбор между доступными нам моделями: LR, DTR и KNN.
В следующем скрипте попробуем нормировать все значения. Так же, попробуем сократить колличество параметров применением метода главных компонент.
Сохраним теперь скорректированные значения в отдельный файл.
"""

df.to_csv(os.path.join(work_dir, 'laptop_price_cleaned.csv'), index=False)
