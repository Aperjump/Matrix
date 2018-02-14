#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <CL/cl.h>
#include <initializer_list>
#include <ctime>
#include <string>
#include "matrix_expr.h"
#include "matrix.h"
#define FILE_NAME "kernel.cl"

int main()
{
	std::ofstream fs("extemp_double_mult_test.csv");
	using namespace ww_matrix;
	ww_clwrapper::_clconfig.init("gpu", 0, 0);
	ww_clwrapper::_clconfig.create_program(FILE_NAME);
	for (size_t size = 0; size <= 5000; size += 500) {
		Matrix<double> a(size, size, 1);
		Matrix<double> b(size, size, 2);
		Matrix<double> c(size, size, 5);
		Matrix<double> d(size, size);
		clock_t begin_time, end_time;
		begin_time = clock();
		d = a + b + c + b + c;
		end_time = clock();
		double gflops = ((double)size / 1000) *((double)size / 1000)*((double)size / 1000) * 2;
		double time = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
		fs << size << "," << time << "," << gflops / time << endl;
		fprintf(stderr, "finish %d\n", size);
	}
	fs.close();
	return 0;
}