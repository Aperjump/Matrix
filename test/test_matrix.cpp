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
#define SIZE 5
#define USE_GPU
template<typename Expr>
void print_all(Expr const & mat);
int main()
{
	using namespace ww_matrix;
	ww_clwrapper::_clconfig.init("gpu", 0, 0);
	ww_clwrapper::_clconfig.create_program(FILE_NAME);
	Matrix<double> a(SIZE, SIZE, 1);
	Matrix<double> b(SIZE, SIZE, 2);
	Matrix<double> c(SIZE, SIZE, 5);
	Matrix<double> d = a + b;
	print_all(d);
	Matrix<double> e = b * c;
	print_all(e);
	Matrix<double> f = a + b + c * e *d;
	print_all(f);
	Matrix<double> g = (a + b)*c + d * f;
	print_all(g);
}
/*
3.00, 3.00, 3.00, 3.00, 3.00,
3.00, 3.00, 3.00, 3.00, 3.00,
3.00, 3.00, 3.00, 3.00, 3.00,
3.00, 3.00, 3.00, 3.00, 3.00,
3.00, 3.00, 3.00, 3.00, 3.00,

50.00, 50.00, 50.00, 50.00, 50.00,
50.00, 50.00, 50.00, 50.00, 50.00,
50.00, 50.00, 50.00, 50.00, 50.00,
50.00, 50.00, 50.00, 50.00, 50.00,
50.00, 50.00, 50.00, 50.00, 50.00,

18753.00, 18753.00, 18753.00, 18753.00, 18753.00,
18753.00, 18753.00, 18753.00, 18753.00, 18753.00,
18753.00, 18753.00, 18753.00, 18753.00, 18753.00,
18753.00, 18753.00, 18753.00, 18753.00, 18753.00,
18753.00, 18753.00, 18753.00, 18753.00, 18753.00,

281370.00, 281370.00, 281370.00, 281370.00, 281370.00,
281370.00, 281370.00, 281370.00, 281370.00, 281370.00,
281370.00, 281370.00, 281370.00, 281370.00, 281370.00,
281370.00, 281370.00, 281370.00, 281370.00, 281370.00,
281370.00, 281370.00, 281370.00, 281370.00, 281370.00,
*/
template<typename Expr>
void print_all(Expr const & mat) {
	for (size_t i = 0; i < mat.rw_size(); i++) {
		for (size_t j = 0; j < mat.cl_size(); j++) {
			printf("%.2f, ", mat(i, j));
		}
		printf("\n");
	}
}
