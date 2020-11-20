// SoftmaxLib.h - Содержит функции для вычисления softmax функции и ее производной
#pragma once

#ifdef SOFTMAXLIB_EXPORTS
#define SOFTMAXLIB_API __declspec(dllexport)
#else
#define SOFTMAXLIB_API __declspec(dllimport)
#endif

// Softmax функция преобразует вектор z размерности N в вектор s той же 
// размерности, где каждая координата s представлена вещественным
// числом в интервале [0, 1] и сумма координат равна 1:
// s(i) = exp(z(i)) / (exp(z(0)) + exp(z(2)) + ... + exp(z(N-1))),
// где i = 0, 1, ..., N-1.
//
// Производная softmax:
// jacobian_m(i, j) = s(j)*(k(i, j) - s(i)), где k(i, j) = { 1, i = j
//														   { 0, иначе

// вычисление softmax функции
// z - входной вектор размерности N
extern "C" SOFTMAXLIB_API double *softmax(int *z, int N);

// вычисление производной softmax функции для исходного вектора z
extern "C" SOFTMAXLIB_API double **softmax_grad(int *z, int N);