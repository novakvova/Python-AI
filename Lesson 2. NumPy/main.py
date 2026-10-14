import numpy as np #Бібліотека для роботи з масивами C++

myArray = np.array([23, 56, 1, 4, 90, 9])
print(myArray) #виводимо масив
print(" -- Working matrix -- ")
myArray2 = np.array([[12, 3, 4], [8,9,5]])
print(myArray2)
print("-- Нулевий масив --")
zeros = np.zeros((3,3))
print(zeros)
print(" -- Масив із одиничок --")
ones = np.ones((3,3))
print(ones)
print("-- Масив із діапазону значень --")
arr = np.arange(0,10, 2)
print(arr)
print("-- Поділ проміжку на частини --")
arr = np.linspace(0,2,10)
print(arr)

print("-- Масив може зберігати будь-які дані --")
myArray = np.array([[2,3,"Сало"], [4,False, 6]])
print(myArray)
print("-- Про масив можна отримати наступні дані --")
print("Shape", myArray.shape) #Розмір масиву
print("ndim", myArray.ndim) #Кількість вимірів у масиві
print("dtype", myArray.dtype) #тип даних у елементах
print("size", myArray.size) #кількість усіх елементів

print("-- Зріз елементів у масиві --")
myList = np.array([34, 56,77 ,3,-6, 32, 345, 12, -8])
print("First element", myList[0])
print("Last element", myList[-1]) #Останій елемент списку
print("1:4", myList[1:4]) # не включно із 4 - вказуємо індекс
print(":3", myList[:3]) # 3 з початку
print("2:", myList[2:]) # від 2 індексу і до кінця
print("::2", myList[::2]) #кожен другий елемент списку

print("-- Зрізи можна робити із 2 мірними масивами --")
my2d = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
print("My 2d", my2d)
print("[1:, :2]", my2d[1:, :2])

print("-- Операції над масивами --")
a = np.array([2,13,24,6,1])
b = np.array([-4,4,5,9,-6])
print("a:", a)
print("b:", b)
print("a+b:", a+b)
print("a-b:", a-b)
print("a*b:", a*b)
print("a/b:", a/b)
print("a//b:", a//b) # ділення без остачі із заокругленням до меншого
print("a%b:", a%b)
# print("a**b:", a**b)
print("a == b :", a == b)
print("a != b :", a != b)

print(" -- Статичні методи у np -- ")
data = np.array([2,3,5,6,7,3,2,90,54,3,4])
print("data", data)
print("mean", np.mean(data)) # Середнє арифтетичне
print("median", np.median(data))  # Медіана більше точне середнє
print("min", np.min(data))
print("max", np.max(data))



