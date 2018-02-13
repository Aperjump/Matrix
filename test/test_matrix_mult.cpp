#include <cstdio>
#include <cstdlib>
#include <CL/cl.h>
#include <initializer_list>
#include <ctime>
#include <string>
#include "matrix_expr.h"
#include "matrix.h"
#define SIZE 3200
#define FILE_NAME "kernel.cl"
int main()
{
	ww_clwrapper::_clconfig.init("gpu", 0, 0);
	ww_clwrapper::_clconfig.create_program(FILE_NAME);
	using namespace ww_matrix;
	Matrix<double> a(SIZE, SIZE, 1);
	Matrix<double> b(SIZE, SIZE, 2);
	Matrix<double> c(SIZE, SIZE, 5);
	Matrix<double> d(SIZE, SIZE);
	clock_t begin_time, end_time;
	begin_time = clock();
	d = a + b;
	end_time = clock();
	// mult and add--> 2*glops
	double gflops = ((double)SIZE / 1000) *((double)SIZE / 1000)*((double)SIZE / 1000) * 2;
	double time = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
	printf("Time spent: %.2f, gflops: %.2f, element is %.2f\n", time, gflops / time, d(1,1));
	return 0;
}