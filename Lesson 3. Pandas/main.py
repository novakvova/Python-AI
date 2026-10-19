import pandas as pd
import numpy as np

# s = pd.Series([20,40,60,80,100])
# print("Series\n",s)

# Можна вказувати свої індекси для елементів
# s = pd.Series([20,40,60], index=['A', 'B', 'C'])
# print("Series index\n",s)

# Запишемо масив у Pandas
# data = {"Київ": 328, "Львів": 210, "Тернопіль": 160}
# s = pd.Series(data)
# print("Словник:\n",s)

#s = pd.Series(12, index=['a', 'b', 'c', 'd'])
# print("Константа\n",s)

# - доступ до даних у Pandas
# s = pd.Series([20,30,60], index=['a','b','c'])
# # print(s[0]) # Буде помилка, бо індекса немає>
# print(s['a']) # Звертаємося до індексу
# print(s.iloc[0]) # за позицією у списку
# print(s[0:2]) # зріз по даних

# Працюємо із табличними даними
# data = {
#     'Ім\'я': ["Іванка","Олеся","Христина"],
#     "Вік": [16, 18, 21],
#     "Місто": ["Луцьк","Рівне","Львів"]
# }
# df = pd.DataFrame(data)
# print(df)

# data = [
#     {"Ім'я": "Іванка", 'Вік': 35, "Місто": "Київ"},
#     {"Ім'я": "Матвій", 'Вік': 25, "Місто": "Рівне"},
#     {"Ім'я": "Юлік", 'Вік': 30, "Місто": "Львів"},
# ]
# df = pd.DataFrame(data)
# print(df)

# data = np.array([[1,2,3],[4,5,6],[7,8,9]])
# df = pd.DataFrame(data,
#                   columns=['A','B','C'],
#                   index = ["рядок 1", "рядок 2", "рядок 3"])
# print(df)

# s1 = pd.Series([1,2,3])
# s2 = pd.Series([4,5,6])
# df = pd.DataFrame({"col1": s1, "col2": s2})
# print(df)

sales_data = {
    'Дата': pd.date_range('2024-01-01', periods=10, freq='D'),
    'Продукт': ['Ноутбук', 'Телефон', 'Планшет', 'Навушники', 'Клавіатура',
                'Миша', 'Монітор', 'Ноутбук', 'Телефон', 'Планшет'],
    'Кількість': [2, 5, 3, 10, 7, 15, 1, 3, 4, 2],
    'Ціна': [25000, 15000, 12000, 2000, 1500, 500, 8000, 25000, 15000, 12000],
    'Регіон': ['Київ', 'Львів', 'Одеса', 'Київ', 'Харків',
               'Львів', 'Київ', 'Одеса', 'Харків', 'Львів']
}

df_sales = pd.DataFrame(sales_data)
print(df_sales)
print()

# Запис у csv
print("== Запис у CSV ==")
df_sales.to_csv('sales.csv', index=False, encoding='utf-8')
print("Збережено у CSV")
print()

#Читання із csv
print("=== Читання CSV ===")
df_sales = pd.read_csv('sales.csv')
print("Прочитані дані з файлу:")
print(df_sales.head(3))
print()
