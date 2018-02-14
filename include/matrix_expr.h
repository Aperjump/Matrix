#pragma once
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
//#include "openCL_wrapper.h"
#define ker1 "core"
#define add "add"
#define CEIL_DIV(x,y) ((x) + (y) - 1) / (y)
#define TS 32
#define WPT 8
#define USE_GPU
//#define PRINT

namespace ww_clwrapper {
	cl_platform_id choose_platform(const std::string& pltfrm_name);
	// offset is device's position in a vector<cl_device_id>
	// this function will call choose_mult_device and select one with offset
	cl_device_id choose_device(cl_platform_id platform, const std::string& devs_name, size_t offset);
	std::vector<cl_device_id> choose_mult_device(cl_platform_id, const std::string& device_type);
	// contains all required structure for computation
	class CL_Base {
	public:
		CL_Base(const std::string& pltform_name) { _platform = ww_clwrapper::choose_platform(pltform_name); }
		CL_Base(cl_platform_id platform) : _platform(platform) { }
		CL_Base(CL_Base& other) : _platform(other._platform),
			current_device(other.current_device), _devices(other._devices),
			_context(other._context), _cmdque(other._cmdque), _program(other._program) { }
		CL_Base& operator= (const CL_Base& other);
		~CL_Base() { }
		bool init(const std::string& device_type, cl_command_queue_properties queue_prop, 
			const cl_context_properties* additional_context_props);
		cl_device_id switch_device(int offset);
		bool create_program(const std::string& file);
		//bool create_kernels();

		cl_platform_id _platform;
		cl_device_id current_device;
		std::vector<cl_device_id> _devices;
		cl_context _context;
		cl_command_queue _cmdque;
		cl_program _program;
		//cl_kernel* _kernels;
		//cl_uint kernel_num;
	};
	static CL_Base _clconfig("CUDA");
}
namespace ww_traits {
	struct _true_type { };
	struct _false_type { };
	template<class T>
	struct __traits
	{
		typedef typename T::_incld_strg _copy_move;
	};
}
namespace ww_matrix
{
	template<typename T>
	class Storage {
	public:
		Storage() = default;
		Storage(size_t row_size, size_t col_size) : _row_size(row_size), _col_size(col_size),
			_buffer_available(false), _val_array(new T[row_size * col_size]) { }
		Storage(size_t row_size, size_t col_size, T val) : _row_size(row_size), _col_size(col_size),
			_buffer_available(false), _val_array(new T[row_size * col_size]) {
			size_t total_size = _row_size * _col_size;
			for (size_t i = 0; i < total_size; i++) {
				_val_array[i] = val;
			}
		}
		~Storage()
		{
			// test whether used buffer
			if (_buffer_available)
				clReleaseMemObject(_buffer);
		}
		Storage(const Storage& other) : _row_size(other._row_size), _col_size(other._col_size),
			_buffer_available(false), _val_array(other._val_array) { }
		Storage(const Storage&& other) : _row_size(other._row_size), _col_size(other._col_size),
			_buffer_available(false), _val_array(other._val_array) { }
		template<typename T2>
		Storage(const T2& other) : _row_size(other.rw_size()), _col_size(other.cl_size()),
			_buffer_available(false), _val_array(new T[_col_size * _col_size])
		{
			for (size_t i = 0; i < _row_size; i++) {
				for (size_t j = 0; j < _col_size; j++) {
					size_t index = i * _col_size + j;
					_val_array[index] = other(i, j);
				}
			}
		}
		Storage<T>& operator= (const Storage<T>& other)
		{
			_row_size = other._row_size;
			_col_size = other._col_size;
			_buffer = other._buffer;
			_val_array = other._val_array;
			return *this;
		}
		cl_mem create_buffer(size_t row, size_t col, int buffer_type) {
			cl_int err = 0;
			_buffer = clCreateBuffer(ww_clwrapper::_clconfig._context,
				buffer_type, row*col * sizeof(T), NULL, &err);
			_buffer_available = true;
			clEnqueueWriteBuffer(ww_clwrapper::_clconfig._cmdque, _buffer,  CL_TRUE, 0, row*col * sizeof(T), _val_array.get(), 0, NULL, NULL);
			if (err < 0)
				return NULL;
			return _buffer;
		}
		void collect_val(size_t row, size_t col, cl_mem& buffer) {
			clEnqueueReadBuffer(ww_clwrapper::_clconfig._cmdque, buffer, CL_TRUE, 0, row*col * sizeof(T), _val_array.get(), 0, NULL, NULL);
		}
		T operator() (size_t i, size_t j) const {
			size_t index = i * _col_size + j;
			return _val_array[index];
		}
		T& operator() (size_t i, size_t j) {
			size_t index = i * _col_size + j;
			return _val_array[index];
		}
		size_t rw_size() const {
			return _row_size;
		}
		size_t cl_size() const {
			return _col_size;
		}
		size_t tt_size() const {
			return _row_size * _col_size;
		}
		cl_mem* buffer() {
			return &_buffer;
		}
		T* get() {
			return _val_array.get();
		}
		mutable cl_mem _buffer;
	private:
		size_t _row_size;
		size_t _col_size;
		std::shared_ptr<T[]> _val_array;
		void copy(std::shared_ptr<T[]> other) {
			T* pointer = other.get();
			size_t total_size = _row_size * _col_size;
			for (size_t i = 0; i < total_size; i++) {
				_val_array[i] = pointer[i];
			}
		}
		bool _buffer_available;
	};
	/* 
	I thought about using CRTP but I think this pattern is 
	mainly used to adding functionality for derived type. 
	But here this pattern may not have much usage.  
	*/
	template<typename E>
	class Matrix_expression {
	public:
		typedef E expression_type;
		const expression_type& operator()() const {
			return *static_cast<const expression_type*>(this);
		}
		expression_type& operator()() {
			return *static_cast<expression_type*>(this);
		}
	};

	template<typename T, typename Expr = Storage<T> >
	class Matrix : Matrix_expression<Matrix<T, Expr>>{
	private:
		template<typename T2, typename Expr2>
		void pass_val(Matrix<T2, Expr2> const & other, ww_traits::_true_type)
		{
			_expr = other.rep().rep();
		}
		//every template parameter name can only instaintiated once
		template<typename T3, typename Expr3>
		void pass_val(Matrix<T3, Expr3> const & other, ww_traits::_false_type)
		{
			size_t row_num = other.rw_size();
			size_t col_num = other.cl_size();
			for (size_t i = 0; i < row_num; i++)
			{
				for (size_t j = 0; j < col_num; j++)
				{
					_expr(i, j) = other(i, j);
				}
			}
		}
	public:
	typedef T val_type;
	typedef Expr expression_type;
		Matrix(size_t row_size, size_t col_size) : _expr(row_size, col_size) { }
		Matrix(size_t row_size, size_t col_size, val_type val) : _expr(row_size, col_size, val) { }
		Matrix(expression_type const& other) : _expr(other) { }
		template<typename T, typename OP>
		Matrix(Matrix<T, OP> const & other) {  _expr = other.rep(); }
		size_t rw_size() const {
			return _expr.rw_size();
		}
		size_t cl_size() const {
			return _expr.cl_size();
		}
		size_t tt_size() const {
			return _expr.rw_size() *  _expr.cl_size();
		}
		T operator() (size_t i, size_t j) const {
			return _expr(i,j);
		}
		T& operator() (size_t i, size_t j) {
			return _expr(i, j);
		}
		Matrix& operator= (const Matrix& other) { _expr = other.rep(); return *this; }
		template<typename T2, typename Expr>
		Matrix& operator= (Matrix<T2, Expr> const & other)
		{
			pass_val(other, ww_traits::__traits<Expr>::_copy_move());
			return *this;
		}

		expression_type* get_pointer() {
			return &_expr;
		}
		expression_type& rep() {
			return _expr;
		}
		expression_type const & rep() const {
			return _expr;
		}
		T* get() {
			return _expr.get();
		}
	private:
		expression_type _expr;
	};
	template<typename T, typename OP1, typename OP2>
	class Matrix_Add :Matrix_expression<Matrix_Add<T, OP1, OP2>> {
	public:
		typedef ww_traits::_false_type _incld_strg;
		Matrix_Add(OP1 const& op1, OP2 const& op2) : _op1(op1), _op2(op2) { }
		T operator() (size_t i, size_t j) const {
			return _op1(i,j)+ _op2(i,j);
		}
		T& operator() (size_t i, size_t j) {
			return _op1(i,j) + _op2(i,j);
		}
		size_t rw_size() const {
			return _op1.rw_size();
		}
		size_t cl_size() const {
			return _op1.cl_size();
		}
		size_t tt_size() const {
			return _op1.rw_size() * _op2.cl_size();
		}
	private:
		OP1 const& _op1;
		OP2 const& _op2;
	};
	template<typename T, typename OP1, typename OP2>
	class Matrix_Add_gpu :Matrix_expression<Matrix_Add_gpu<T, OP1, OP2>> {
	public:
		typedef ww_traits::_true_type _incld_strg;
		Matrix_Add_gpu(OP1 const & op1, OP2 const & op2) : _lhs(op1),
			_rhs(op2), _ret(op1.rw_size(), op2.cl_size())
		{
			assert(op1.cl_size() == op2.rw_size());
			size_t M = _lhs.rw_size();
			size_t N = _rhs.cl_size();
			cl_int err = 0;
			Storage<T>* lhs_expr = _lhs.get_pointer();
			Storage<T>* rhs_expr = _rhs.get_pointer();
			Storage<T>* ret_expr = _ret.get_pointer();
			cl_mem lhs_buffer = lhs_expr->create_buffer(M, N, CL_MEM_READ_ONLY);
			cl_mem rhs_buffer = rhs_expr->create_buffer(M, N, CL_MEM_READ_ONLY);
			cl_mem ret_buffer = ret_expr->create_buffer(M, N, CL_MEM_WRITE_ONLY);
			cl_kernel Kernel = clCreateKernel(ww_clwrapper::_clconfig._program, add, &err);
			clSetKernelArg(Kernel, 0, sizeof(int), (void*)&M);
			clSetKernelArg(Kernel, 1, sizeof(int), (void*)&N);
			clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&(lhs_buffer));
			clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&(rhs_buffer));
			clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*)&(ret_buffer));
			const size_t global[2] = { M, N };
			cl_event event = NULL;
			clEnqueueNDRangeKernel(ww_clwrapper::_clconfig._cmdque, Kernel, 2, NULL, global, NULL, 0, NULL, &event);
			ret_expr->collect_val(M, N, ret_buffer);
			clReleaseKernel(Kernel);
		}
		T operator() (size_t i, size_t j) const {
			return _ret(i, j);
		}
		T& operator() (size_t i, size_t j) {
			return _ret(i, j);
		}
		size_t rw_size() const {
			return _lhs.rw_size();
		}
		size_t cl_size() const {
			return _rhs.cl_size();
		}
		size_t tt_size() const {
			return _lhs.rw_size() * _rhs.cl_size();
		}
		Storage<T>& rep()
		{
			return _ret.rep();
		}
		Storage<T> const& rep() const
		{
			return _ret.rep();
		}
	private:
		Matrix<T>  _lhs;
		Matrix<T>  _rhs;
		Matrix<T> _ret;
	};
	template<typename T, typename OP1, typename OP2>
	class Matrix_Mult_gpu : Matrix_expression<Matrix_Mult_gpu<T, OP1, OP2>> {
	public:
		typedef ww_traits::_true_type _incld_strg;
		Matrix_Mult_gpu(OP1 const & op1, OP2 const & op2) : _lhs(op1),
			_rhs(op2), _ret(op1.rw_size(), op2.cl_size())
		{
			assert(op1.cl_size() == op2.rw_size());
			size_t M = _lhs.rw_size();
			size_t N = _rhs.cl_size();
			size_t K = _lhs.cl_size();
			cl_int err = 0;
			Storage<T>* lhs_expr = _lhs.get_pointer();
			Storage<T>* rhs_expr = _rhs.get_pointer();
			Storage<T>* ret_expr = _ret.get_pointer();
			cl_mem lhs_buffer = lhs_expr->create_buffer(M, K, CL_MEM_READ_ONLY);
			cl_mem rhs_buffer = rhs_expr->create_buffer(K, N, CL_MEM_READ_ONLY);
			cl_mem ret_buffer = ret_expr->create_buffer(M, N, CL_MEM_WRITE_ONLY);
			size_t M_Ex = CEIL_DIV(M, TS) * TS;
			size_t N_Ex = CEIL_DIV(N, TS) * TS;
			size_t K_Ex = CEIL_DIV(N, TS) * TS;
			cl_mem lhs_pad = clCreateBuffer(ww_clwrapper::_clconfig._context, CL_MEM_READ_ONLY, M_Ex * K_Ex * sizeof(T), NULL, NULL);
			cl_mem rhs_pad = clCreateBuffer(ww_clwrapper::_clconfig._context, CL_MEM_READ_ONLY, K_Ex * N_Ex * sizeof(T), NULL, NULL);
			cl_mem ret_pad = clCreateBuffer(ww_clwrapper::_clconfig._context, CL_MEM_READ_ONLY, M_Ex * N_Ex * sizeof(T), NULL, NULL);
			// padding matrix lhs
			cl_kernel padding_a = clCreateKernel(ww_clwrapper::_clconfig._program, "padding", &err);
			err = clSetKernelArg(padding_a, 0, sizeof(int), (void*)&M);
			err = clSetKernelArg(padding_a, 1, sizeof(int), (void*)&K);
			err = clSetKernelArg(padding_a, 2, sizeof(int), (void*)&M_Ex);
			err = clSetKernelArg(padding_a, 3, sizeof(int), (void*)&(K_Ex));
			err = clSetKernelArg(padding_a, 4, sizeof(cl_mem), (void*)&(lhs_buffer));
			err = clSetKernelArg(padding_a, 5, sizeof(cl_mem), (void*)&(lhs_pad));
			// padding matrix rhs
			cl_kernel padding_b = clCreateKernel(ww_clwrapper::_clconfig._program, "padding", &err);
			err = clSetKernelArg(padding_b, 0, sizeof(int), (void*)&K);
			err = clSetKernelArg(padding_b, 1, sizeof(int), (void*)&N);
			err = clSetKernelArg(padding_b, 2, sizeof(int), (void*)&K_Ex);
			err = clSetKernelArg(padding_b, 3, sizeof(int), (void*)&(N_Ex));
			err = clSetKernelArg(padding_b, 4, sizeof(cl_mem), (void*)&(rhs_buffer));
			err = clSetKernelArg(padding_b, 5, sizeof(cl_mem), (void*)&(rhs_pad));
			// core function
			cl_kernel Kernel = clCreateKernel(ww_clwrapper::_clconfig._program, "core", &err);
			err = clSetKernelArg(Kernel, 0, sizeof(int), (void*)&M_Ex);
			err = clSetKernelArg(Kernel, 1, sizeof(int), (void*)&N_Ex);
			err = clSetKernelArg(Kernel, 2, sizeof(int), (void*)&K_Ex);
			err = clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&(lhs_pad));
			err = clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*)&(rhs_pad));
			err = clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*)&(ret_pad));
			// return padding matrix
			cl_kernel remove_c = clCreateKernel(ww_clwrapper::_clconfig._program, "return_padding", &err);
			err = clSetKernelArg(remove_c, 0, sizeof(int), (void*)&M_Ex);
			err = clSetKernelArg(remove_c, 1, sizeof(int), (void*)&N_Ex);
			err = clSetKernelArg(remove_c, 2, sizeof(int), (void*)&M);
			err = clSetKernelArg(remove_c, 3, sizeof(int), (void*)&N);
			err = clSetKernelArg(remove_c, 4, sizeof(cl_mem), (void*)&(ret_pad));
			err = clSetKernelArg(remove_c, 5, sizeof(cl_mem), (void*)&(ret_buffer));

			const size_t local[2] = { TS, TS };
			const size_t global[2] = { M_Ex, N_Ex };
			const size_t local_2[2] = { TS, TS / WPT };
			const size_t global_2[2] = { M_Ex, N_Ex / WPT };
			const size_t lhs_Ex_global[2] = { M_Ex, K_Ex };
			const size_t rhs_Ex_global[2] = { K_Ex, N_Ex };
			const size_t ret_Ex_global[2] = { M_Ex, N_Ex };
			cl_event event = NULL;
			err = clEnqueueNDRangeKernel(ww_clwrapper::_clconfig._cmdque, padding_a, 2, NULL, lhs_Ex_global, local, 0, NULL, &event);
			err = clEnqueueNDRangeKernel(ww_clwrapper::_clconfig._cmdque, padding_b, 2, NULL, rhs_Ex_global, local, 0, NULL, &event);
			err = clEnqueueNDRangeKernel(ww_clwrapper::_clconfig._cmdque, Kernel, 2, NULL, global_2, local_2, 0, NULL, &event);
			//T* test = (T*)malloc(M_Ex * N_Ex * sizeof(T));
			//clEnqueueReadBuffer(ww_clwrapper::_clconfig._cmdque, ret_pad, CL_TRUE, 0, M_Ex * N_Ex * sizeof(T), test, 0, NULL, NULL);
			//for (size_t i = 0; i < M_Ex; i++) {
			//	for (size_t j = 0; j < N_Ex; j++) {
			//		printf("%.1f, ", test[i * N_Ex + j]);
			//	}
			//	printf("\n");
			//}
			err = clEnqueueNDRangeKernel(ww_clwrapper::_clconfig._cmdque, remove_c, 2, NULL, ret_Ex_global, local, 0, NULL, &event);

			ret_expr->collect_val(M, N, ret_buffer);
			clReleaseKernel(Kernel);
		}
		T operator() (size_t i, size_t j) const {
			return _ret(i, j);
		}
		T& operator() (size_t i, size_t j) {
			return _ret(i, j);
		}
		size_t rw_size() const {
			return _lhs.rw_size();
		}
		size_t cl_size() const {
			return _rhs.cl_size();
		}
		size_t tt_size() const {
			return _lhs.rw_size() * _rhs.cl_size();
		}
		Storage<T>& rep()
		{
			return _ret.rep();
		}
		Storage<T> const& rep() const
		{
			return _ret.rep();
		}
	private:
		Matrix<T>  _lhs;
		Matrix<T>  _rhs;
		Matrix<T> _ret;
	};
#ifdef USE_GPU
	template<typename T, typename OP1, typename OP2>
	Matrix<T, Matrix_Add_gpu<T, OP1, OP2>> operator + (Matrix<T, OP1> const & lhs,
		 Matrix<T, OP2> const & rhs) {
		assert(lhs.rw_size() == rhs.rw_size() && lhs.cl_size() == rhs.cl_size());
#ifdef PRINT
		printf("Use Matrix_Add_gpu\n");
#endif
		return Matrix<T, Matrix_Add_gpu<T, OP1, OP2>>(Matrix_Add_gpu<T, OP1, OP2>(lhs.rep(), rhs.rep()));
	}
#else
	template<typename T, typename OP1, typename OP2>
	Matrix<T, Matrix_Add<T, OP1, OP2>> operator + (Matrix<T, OP1> const & lhs,
		Matrix<T, OP2> const & rhs) {
		assert(lhs.rw_size() == rhs.rw_size() && lhs.cl_size() == rhs.cl_size());
#ifdef PRINT
		printf("Use Matrix_Add\n");
#endif
		return Matrix<T, Matrix_Add<T, OP1, OP2>>(Matrix_Add<T, OP1, OP2>(lhs.rep(), rhs.rep()));
	}
#endif
	template<typename T, typename OP1, typename OP2>
	Matrix<T, Matrix_Mult_gpu<T, OP1, OP2>> operator * (Matrix<T, OP1> const & lhs,
		Matrix<T, OP2> const & rhs) {
		assert(lhs.cl_size() == rhs.rw_size());
		return Matrix<T, Matrix_Mult_gpu<T, OP1, OP2>>(Matrix_Mult_gpu<T, OP1, OP2>(lhs.rep(), rhs.rep()));
	}
};
namespace ww_simple_matrix {
	template<typename T>
	class Storage {
	public:
		Storage() = default;
		Storage(size_t row, size_t col) : _row_size(row), _col_size(col), _storage(row*col, 0) { }
		Storage(size_t row, size_t col, T val) :_row_size(row), _col_size(col), _storage(row*col, val) { }
		T operator() (size_t i, size_t j) const {
			size_t index = i * _col_size + j;
			return _storage[index];
		}
		T& operator() (size_t i, size_t j) {
			size_t index = i * _col_size + j;
			return _storage[index];
		}
		void push_back(T val) {
			_storage.push_back(val);
		}
		void resize(size_t row, size_t col, T val) {
			_row_size = row;
			_col_size = col;
			_storage.resize(row*col, val);
		}
		size_t rw_size() const { return _row_size; }
		size_t cl_size() const { return _col_size; }
	private:
		size_t _row_size;
		size_t _col_size;
		std::vector<T> _storage;
	};
	template<typename T, typename Expr = Storage<T>>
	class SMatrix {
	public:
		SMatrix() = default;
		SMatrix(size_t row, size_t col) : _row_size(row), _col_size(col),
			_expr(row, col) { }
		SMatrix(size_t row, size_t col, T val) : _row_size(row), _col_size(col),
			_expr(row, col, val) { }
		SMatrix(std::initializer_list<std::initializer_list<T> > tmp_list): _row_size(tmp_list.size()),
			_col_size((*tmp_list.begin()).size()), _expr(_row_size, _col_size){
			int i = 0;
			for (auto iter = tmp_list.begin(); iter != tmp_list.end(); iter++)
			{
				int j = 0;
				for (auto iter2 = iter->begin(); iter2 != iter->end(); iter2++)
				{
					_expr(i, j) = *iter2;
					j++;
				}
				i++;
			}
		}
		SMatrix(SMatrix<T, Expr> const & other){ 
			_row_size = other.rw_size();
			_col_size = other.cl_size();
			_expr = other.rep();
		}
		template<typename T2, typename Expr2>
		SMatrix(SMatrix<T2, Expr2> const & other)
		{
			_row_size = other.rw_size();
			_col_size = other.cl_size();
			_expr.resize(_row_size, _col_size, 0);
			for (size_t i = 0; i < _row_size; i++) {
				for (size_t j = 0; j < _col_size; j++) {
					_expr(i, j) = other(i, j);
				}
			}
		}
		SMatrix(Expr const & other) : _row_size(other.rw_size()), 
			_col_size(other.cl_size()), _expr(other){  }
		SMatrix& operator= (const SMatrix& other) { 
			_row_size = other.rw_size();
			_col_size = other.cl_size();
			_expr = other.rep();
			return *this;
		}
		template<typename T2, typename Expr2>
		SMatrix& operator= (SMatrix<T2, Expr2> const & other)
		{
			
			_row_size = other.rw_size();
			_col_size = other.cl_size();
			_expr.resize(_row_size, _col_size, 0);
			for (size_t i = 0; i < _row_size; i++) {
				for (size_t j = 0; j < _col_size; j++) {
					_expr(i,j) = other(i, j);
				}
			}
			return *this;
		}
		size_t rw_size() const { return _row_size;  }
		size_t cl_size() const { return _col_size;  }
		T operator() (size_t i, size_t j) const {
			return _expr(i,j);
		}
		T& operator() (size_t i, size_t j) {
			return _expr(i,j);
		}
		Expr& rep() {
			return _expr;
		}
		Expr const & rep() const {
			return _expr;
		}
	private:
		size_t _row_size;
		size_t _col_size;
		Expr _expr;
	};
	template<typename T, typename Expr1, typename Expr2>
	class SMatrix_Add {
	public:
		SMatrix_Add(Expr1 const & lhs, Expr2 const & rhs) : _lhs(lhs), _rhs(rhs),_row_size(lhs.rw_size()),
		_col_size(rhs.cl_size()){	}
		SMatrix_Add& operator=(SMatrix_Add<T, Expr1, Expr2> const & other) {
			_lhs = other._lhs;
			_rhs = other._rhs;
			return *this;
		}
		T operator() (size_t i, size_t j) const {
			return _lhs(i,j)+ _rhs(i,j);
		}
		T& operator() (size_t i, size_t j) {
			return _lhs(i, j) + _rhs(i, j);
		}
		size_t rw_size() const { return _row_size; }
		size_t cl_size() const { return _col_size; }
	private:
		size_t _row_size;
		size_t _col_size;
		Expr1 const & _lhs;
		Expr2 const & _rhs;
	};
	template<typename T, typename Expr1, typename Expr2>
	class SMatrix_Mul {
	public:
		SMatrix_Mul(Expr1 const & lhs, Expr2 const & rhs) : _lhs(lhs), _rhs(rhs), _row_size(lhs.rw_size()),
			_col_size(rhs.cl_size()) {	}
		T operator() (size_t i, size_t j) const {
			T ret = 0;
			for (size_t k = 0; k < _lhs.cl_size(); k++)
			{
				ret += _lhs(i, k) * _rhs(k, j);
			}
			return ret;
		}
		T& operator() (size_t i, size_t j) {
			return _lhs(i, j) + _rhs(i, j);
		}
		size_t rw_size() const { return _row_size; }
		size_t cl_size() const { return _col_size; }
	private:
		size_t _row_size;
		size_t _col_size;
		Expr1 const & _lhs;
		Expr2 const & _rhs;
	};
	template<typename T, typename Expr1, typename Expr2>
	SMatrix<T, SMatrix_Add<T, Expr1, Expr2>> operator + (SMatrix<T, Expr1> const & lhs,
		SMatrix<T, Expr2> const & rhs) {
		assert(lhs.rw_size() == rhs.rw_size() && lhs.cl_size() == rhs.cl_size());
		return SMatrix<T, SMatrix_Add<T, Expr1, Expr2>>(SMatrix_Add<T, Expr1, Expr2>(lhs.rep(), rhs.rep()));
	}
	template<typename T, typename Expr1, typename Expr2>
	SMatrix<T, SMatrix_Mul<T, Expr1, Expr2>> operator * (SMatrix<T, Expr1> const & lhs,
		SMatrix<T, Expr2> const & rhs) {
		assert(lhs.cl_size() == rhs.rw_size());
		return SMatrix<T, SMatrix_Mul<T, Expr1, Expr2>>(SMatrix_Mul<T, Expr1, Expr2>(lhs.rep(), rhs.rep()));
	}
}