#include "pch.h"
#include <cmath>
#include "SoftmaxLib.h"

// ���������� softmax �������
// z - ������� ������ ����������� N
double *softmax(int *z, int N)
{
	// ���������� ���������� ������������
	double sum = 0;

	// ���������� ����� ��������� �� ��������� ������� z
	for (int i = 0; i < N; ++i)
	{
		sum += exp(z[i]);
	}
	
	// s - ������ ����������� N
	double *s = new double[N];
	
	// ���������� ������� s
	for (int i = 0; i < N; ++i)
	{
		s[i] = exp(z[i]) / sum;
	}
	return s;
}

// ���������� ����������� softmax ������� ��� ��������� ������� z
double **softmax_grad(int *z, int N)
{
	// s - �������� softmax ������� �� ������� z
	double *s = softmax(z, N);

	// jacobian_m - ������� n*n
	double **jacobian_m = new double*[N];
	for (int i = 0; i < N; ++i) jacobian_m[i] = new double[N];

	// ���������� ������� jacobian_m
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			jacobian_m[i][j] = s[j] * ((i == j) - s[i]);
		}
	}
	return jacobian_m;
}