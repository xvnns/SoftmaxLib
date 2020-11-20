Вариант 36 – Расчёт производной softmax функции.

# SoftmaxLib

Реализация динамической библиотеки SoftmaxLib для вычисления производной softmax функции

Softmax функция преобразует вектор z размерности N в вектор s той же размерности, где каждая координата  
вектора s представлена вещественным числом в интервале [0, 1] и сумма координат равна 1:  
s(i) = exp(z(i)) / (exp(z(0)) + exp(z(2)) + ... + exp(z(N-1))), где i = 0, 1, ..., N-1.

Производная softmax:
jacobian_m(i, j) = s(j)(k(i, j) - s(i)), где jacobian_m(i, j) - двухмерная матрица;  
k(i, j) = { 1, i = j  
&emsp;&emsp;&emsp;{ 0, иначе
