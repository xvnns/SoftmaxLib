// SoftmaxLib.h - �������� ������� ��� ���������� softmax ������� � �� �����������
#pragma once

#ifdef SOFTMAXLIB_EXPORTS
#define SOFTMAXLIB_API __declspec(dllexport)
#else
#define SOFTMAXLIB_API __declspec(dllimport)
#endif

// Softmax ������� ����������� ������ z ����������� N � ������ s ��� �� 
// �����������, ��� ������ ���������� s ������������ ������������
// ������ � ��������� [0, 1] � ����� ��������� ����� 1:
// s(i) = exp(z(i)) / (exp(z(0)) + exp(z(2)) + ... + exp(z(N-1))),
// ��� i = 0, 1, ..., N-1.
//
// ����������� softmax:
// jacobian_m(i, j) = s(j)*(k(i, j) - s(i)), ��� k(i, j) = { 1, i = j
//														   { 0, �����

// ���������� softmax �������
// z - ������� ������ ����������� N
extern "C" SOFTMAXLIB_API double *softmax(int *z, int N);

// ���������� ����������� softmax ������� ��� ��������� ������� z
extern "C" SOFTMAXLIB_API double **softmax_grad(int *z, int N);