import numpy as np
import random
import matplotlib.pyplot as plt


def cart2pol(x, y):
  rho = np.sqrt(x**2 + y**2)
  phi = np.arctan2(y, x)
  return (round(rho, 3), round(phi, 3))


def pol2cart(rho, phi):
  x = rho * np.cos(phi)
  y = rho * np.sin(phi)
  return (x, y)


# Обучающая выборка содержит данные обучающего примеров, в каждой строке записаны 4 числа:
# координаты вектора в полярной системе координат (переменные ρ и φ) и в декартовой системе (переменные x и y).
# Всего выборка содержит 100 примеров, в которых вектора расположены в первом квадрате и имеют длину от 0.05 до 1.



def toFixed(numObj, digits=0):
  return f"{numObj:.{digits}f}"


def trainingSetCreateFunction(sizeSet=100):
  coordinatesSet = []
  while (sizeSet > 0):

    VectorLengthCorrect = False
    while (VectorLengthCorrect == False):
      x = round(random.random() * (0.99 - 0.05) + 0.05, 3)
      y = round(random.random() * (0.99 - 0.05) + 0.05, 3)
      if (np.sqrt(x * x + y * y) >= 0.05 and np.sqrt(x * x + y * y) <= 1):
        VectorLengthCorrect = True

    pol = cart2pol(x, y)
    newSet = [pol[0], pol[1], x, y]
    coordinatesSet.append(newSet)
    sizeSet -= 1
  return coordinatesSet


# Функция создания графика точек обучающей выборки в декартовых координатах
def plotTrainingSetPoints(trainingSet):
  x = []
  y = []

  for set in trainingSet:
    x.append(set[2])
    y.append(set[3])

  fig, ax = plt.subplots()

  ax.scatter(x, y, c='green')  #  цвет точек

  ax.set_facecolor('white')  #  цвет области Axes
  ax.set_title('Точки для обучения')  #  заголовок для Axes

  fig.set_figwidth(8)  #  ширина и
  fig.set_figheight(8)  #  высота "Figure"

  plt.savefig('trainingPoints.png')


#активационная функция нейрона relu



def relu(x):
  return np.maximum(x, 0)


#активационная функция нейрона sigmoid
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
  return x * (1 - x)

#сумматорная функция нейрона
def linear(x):
   return x
def d_linear(x):
    return 1

def softplus(x):
    return np.log(1+np.exp(x))
def d_softplus(x):
    return (1/(1+np.exp(-x)))

def summaryFunction(W, x, b):
  print('W:', W)
  mul = np.dot(W.T, x)
  print('Mul:', mul)
  #summary = np.sum(mul, axis = 1, keepdims = True)
  #print('Sum:', summary)
  return (mul)  #+ b)


#print('W1*X:', W1*x)
#mul = W1 * x + b1
#print('W1*X + b1:', for_sum)
#print(np.sum(for_sum, axis = 1, keepdims = True))


#среднеквадратичная ошибка
def mean_squared_error(y_pred, y_true):
  return ((y_pred - y_true)**2).sum() / (2 * y_pred.size)


#точность
def accuracy(y_pred, y_true):
  acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
  return acc.mean()


# функция результат который является сумматорное значение нейрона на слое

#Инициализая параметров нейроной сети:


#for set in trainingSet:
#  print(set)

input_count = 2
output_count = 2
hiddenL_count = int(input('Количество нейронов на скрытом слое:'))
learning_rate = 0.01

a_funcs = [sigmoid,linear]
d_a_funcs = [d_sigmoid,d_linear]

trainingSet = trainingSetCreateFunction()
plotTrainingSetPoints(trainingSet)



W1 = np.random.randn(hiddenL_count,
                     input_count)  #В строке матрицы веса входящие в
#нейрон на h-слое
#print(W1)
#print(W1.shape)
b1 = np.ones((hiddenL_count, 1), 'float')
W2 = np.random.randn(output_count, hiddenL_count)
b2 = np.ones((output_count, 1), 'float')
nr_correct = 0
w_set = [W1,W2]
b_set = (b1,b2)
l_outputs = [0,0]

#print('W1:',W1)
#print('W2:',W2)
#print('b1', b1)
#print('b2', b2)

def f_prop(x):
  i = 0
  io_layer = x
  for w in w_set:
    l_sum = b_set[i] + w @ io_layer
    l_act = a_funcs[i](l_sum)
    io_layer = l_act
    l_outputs[i] = io_layer
    i += 1
  return io_layer

series_acc = 0
epochs = int(input('Введите количество итераций(эпох):'))
testingSet = trainingSetCreateFunction(100)
x_analyze = []
y_analyze = []
y1_analyze = []
sum_e  = 0
for epoch in range(epochs):
  #trainingSet = trainingSetCreateFunction(100)
  for set in trainingSet:
    i = np.random.randint(0, len(trainingSet))
    x = np.array([trainingSet[i][0],
                  trainingSet[i][1]]).reshape(2, 1)  #входной вектор
    y = np.array([trainingSet[i][2],
                  trainingSet[i][3]]).reshape(2, 1)  #вектор правильного ответа


    o = f_prop(x)
    #cost / error calculation
    e = 1 / len(o) * np.sum((o - y)**2, axis=0)
    sum_e += e
    nr_correct += int(np.argmax(o) == np.argmax(y) and np.argmin(o) == np.argmin(y))
    #print(o, y)
    #bp output -> hidden
    delta_o = (o - y) * d_a_funcs[1](o)   #error
    W2 += -learning_rate * delta_o @ np.transpose(l_outputs[0])
    b2 += -learning_rate * delta_o
    #bp hidden -> input
    delta_h = np.transpose(W2) @ delta_o * d_a_funcs[0](l_outputs[0])
    W1 += -learning_rate * delta_h @ np.transpose(x)
    b1 += -learning_rate * delta_h
    w_set = [W1,W2]
    b_set = [b1,b2]

#вычисление точности на тестируемой выборке после каждой итерации(эпохи)
  t_correct = 0
  for t_set,trainSet in zip(testingSet,trainingSet):
    x_test = np.array([t_set[0],t_set[1]]).reshape(2, 1)
    y_test = np.array([t_set[2],t_set[3]]).reshape(2, 1)
    o = f_prop(x_test)
    t_correct += int(np.argmax(o) == np.argmax(y_test))
  x_analyze.append(epoch+1)
  y_analyze.append(round((t_correct/100)*100,2))
  y1_analyze.append(sum_e)




  if(epochs>100):
    if ((epoch + 1) % 10 == 0):
      print(f"{epoch+1} .Acc: {round((nr_correct / 100) * 100, 2)}%")
  else:
    print(f"{epoch+1} .Acc: {round((nr_correct / 100) * 100, 2)}%")

  nr_correct = 0
  sum_e = 0

#построение графика для анализа (x = номер итерации(эпохи), y = точность)
fig1, ax1 = plt.subplots()
ax1.plot(x_analyze, y_analyze, y1_analyze)
ax1.set_title(f'a_func: {a_funcs[0].__name__,a_funcs[1].__name__}, learning_rate: {learning_rate}, h_count: {hiddenL_count}')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy %')
plt.savefig('accuracy_epoch.png')



x = np.array([trainingSet[0][0],
              trainingSet[0][1]]).reshape(2, 1)  #входной вектор
y = np.array([trainingSet[0][2],
              trainingSet[0][3]]).reshape(2, 1)  #вектор правильного ответа
print('x:', x)
print('y:', y)

while (True):
  ro_value = float(input('Введите значение ρ:'))
  fi_value = float(input('Введите значение φ:'))

  x = np.array([[ro_value], [fi_value]])
  print(x)

  o = f_prop(x)
  #print(o)
  ot = np.transpose(o)
  print(f'NN: Polar ({ro_value},{fi_value}) -> Decart {round(ot[0][0],2), round(ot[0][1],2)}')
  xy = pol2cart(ro_value, fi_value)
  print(f'TrueValue: Polar({ro_value},{fi_value}) -> Decart {round(xy[0],2),round(xy[1],2)}')
  #break

