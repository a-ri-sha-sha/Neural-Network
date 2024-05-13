# Neural-Network
Эта библиотека, написанная на языке C++, предоставляет возможность к созданию и обучению полносвязных нейронных сетей. 

## Работа с библиотекой
Для использования библиотечного кода, необходимо подключить один заголовочный файл:
```cpp
#include "Network/net.h"
```
Библиотека предоставляет доступ к 
1. Созданию объекта Network с параметрами:

    • Размеры слоев

    • Нелинейная функция для каждого слоя
```cpp
ActivationFunction f1 = Sigmoid();
ActivationFunction f2 = Tanh();
ActivationFunction f3 = ReLu();
ActivationFunction f4 = LeakyReLu(0.01);
```
2. Тренеровки данного объекта с параметрами:

   • Данные
```cpp
struct Data {
    Matrix input;
    Matrix output;
};
```

   • Количество эпох
```cpp
int epochs;
```
   
   • Желаемая погрешность (рассчитывается функцией потери)
```cpp
double eps;
```
   • Размер батча с данными
   • Желаемая погрешность (рассчитывается функцией потери)
```cpp
Index batch_size;
```
   • Функция потерь
```cpp
LossFunction f1 = MSE();
LossFunction f2 = BCELoss();
```
   • Степень learning_rate. 
   ```cpp
int power_learning_rate = 1;
```
Сам learning_rate вычисляется по формуле 
```cpp
double learning_rate = 1.0 / (1 + std::pow(epoch, power_learning_rate));
```
3. Предсказанию результата на данных
```cpp
Vector Predict(const Matrix &x);
```
## Требования
1. Язык программирования: С++17 (или выше).
2. Система сборки: Cmake 3.24. 
2. Система поддержки версий: Git 2.34.
3. Библиотеки: Eigen 3.4.0, EigenRand
## Пример использования