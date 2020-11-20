#include "pch.h"
#include <cmath>
#include "SoftmaxLib.h"

// вычисление softmax функции
// z - входной вектор размерности N
double *softmax(int *z, int N)
{
	// объ€вление переменной суммировани€
	double sum = 0;

	// вычисление суммы экспонент по значени€м вектора z
	for (int i = 0; i < N; ++i)
	{
		sum += exp(z[i]);
	}
	
	// s - вектор размерности N
	double *s = new double[N];
	
	// заполнение вектора s
	for (int i = 0; i < N; ++i)
	{
		s[i] = exp(z[i]) / sum;
	}
	return s;
}

// вычисление производной softmax функции дл€ исходного вектора z
double **softmax_grad(int *z, int N)
{
	// s - значение softmax функции от вектора z
	double *s = softmax(z, N);

	// jacobian_m - матрица n*n
	double **jacobian_m = new double*[N];
	for (int i = 0; i < N; ++i) jacobian_m[i] = new double[N];

	// заполнение матрицы jacobian_m
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			jacobian_m[i][j] = s[j] * ((i == j) - s[i]);
		}
	}
	return jacobian_m;
}