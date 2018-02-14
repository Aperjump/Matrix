#include <cstdio>
#include <cstdlib>
#include <CL/cl.h>
#include <initializer_list>
#include <ctime>
#include <string>
#include "matrix_expr.h"
#define SIZE 3200
#define FILE_NAME "kernel.cl"
#define USE_GPU
template<typename Expr>
void print_all(Expr const & mat);
int main()
{
	using namespace ww_simple_matrix;
	SMatrix<double> a = { { 1,2,3 },{ 2,3,4 },{ 4,5,6 } };
	SMatrix<double> b = { { 1,2,3 },{ 2,3,4 },{ 4,5,6 } };
	SMatrix<double> c(3, 3, 2);
	SMatrix<double> d(3, 3, 5);
	SMatrix<double> e;
	print_all(a);
	printf("\n");
	e = a + b;
	print_all(e);
	printf("\n");
	e = a * b;
	print_all(e);
	printf("\n");
	e = a + b + c + d;
	print_all(e);	
	printf("\n");
	e = a * b * c * d;
	print_all(e);
	printf("\n");
	SMatrix<double> f(3, 2, 7);
	SMatrix<double> g(2, 5, 4);
	SMatrix<double> h = f * g;
	print_all(h);
	return 0;
}
/*
1.00, 2.00, 3.00,
2.00, 3.00, 4.00,
4.00, 5.00, 6.00,

2.00, 4.00, 6.00,
4.00, 6.00, 8.00,
8.00, 10.00, 12.00,

17.00, 23.00, 29.00,
24.00, 33.00, 42.00,
38.00, 53.00, 68.00,

9.00, 11.00, 13.00,
11.00, 13.00, 15.00,
15.00, 17.00, 19.00,

2070.00, 2070.00, 2070.00,
2970.00, 2970.00, 2970.00,
4770.00, 4770.00, 4770.00,

56.00, 56.00, 56.00, 56.00, 56.00,
56.00, 56.00, 56.00, 56.00, 56.00,
56.00, 56.00, 56.00, 56.00, 56.00,

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
