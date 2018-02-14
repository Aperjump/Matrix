#include "matrix_expr.h"
namespace ww_clwrapper {
	cl_platform_id choose_platform(const std::string& pltfrm_name) {
		cl_uint platform_num = 0;
		cl_int err = clGetPlatformIDs(0, NULL, &platform_num);
		if (err < 0) {
			perror("query platform error");
		}
		std::vector<cl_platform_id> platforms(platform_num);
		err = clGetPlatformIDs(platform_num, &platforms[0], NULL);
		if (err < 0) {
			perror("get platform vector error");
		}
		cl_uint index = platform_num;
		// Query info for each devices and try to match pltfrm_name
		for (cl_uint i = 0; i < platform_num; i++)
		{
			size_t plt_name_len = 0;
			err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, 0, &plt_name_len);
			if (err < 0) {
				perror("cannot find platform");
			}
			std::vector<char> plt_name(plt_name_len);
			err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, plt_name_len,
				&plt_name[0], NULL);
			if (err < 0) {
				perror("cannot find platform");
			}
			if (std::string(&plt_name[0]).find(pltfrm_name) != std::string::npos)
			{
				printf("%s\n", &plt_name[0]);
				index = i;
			}
		}
		if (index >= platform_num) {
			perror("out of platform index\n");
			exit(1);
		}
		// may have truble
		return platforms[index];
	}
	std::vector<cl_device_id> choose_mult_device(cl_platform_id platform, const std::string& device_type) {
		cl_device_type type;
		if (device_type == "gpu" || device_type == "GPU") {
			type = CL_DEVICE_TYPE_GPU;
		}
		else if (device_type == "cpu" || device_type == "CPU") {
			type = CL_DEVICE_TYPE_CPU;
		}
		else {
			type = CL_DEVICE_TYPE_DEFAULT;
		}
		cl_uint device_num = 0;
		cl_int err = clGetDeviceIDs(platform, type, 0, 0, &device_num);
		if (err < 0) {
			perror("cannot find device");
		}
		std::vector<cl_device_id> devices(device_num);
		err = clGetDeviceIDs(platform, type, device_num, &devices[0], NULL);
		if (err < 0) {
			perror("cannot find device");
		}
		return devices;
	}
	cl_device_id choose_device(cl_platform_id platform, const std::string& devs_name, size_t offset) {
		std::vector<cl_device_id> devices = choose_mult_device(platform, devs_name);
		assert(offset < devices.size());
		return devices[offset];
	}
	CL_Base& CL_Base::operator= (const CL_Base& other) {
		_platform = other._platform;
		current_device = other.current_device;
		_devices = other._devices;
		_context = other._context;
		_cmdque = other._cmdque;
		_program = other._program;
		return *this;
	}
	bool CL_Base::init(const std::string& device_type, cl_command_queue_properties queue_prop = 0,
		const cl_context_properties* additional_context_props = 0) {
		cl_int err = 0;
		_devices = ww_clwrapper::choose_mult_device(_platform, device_type);
		current_device = _devices[0];
		char deviceName[1024];
		clGetDeviceInfo(current_device, CL_DEVICE_NAME, 1024, deviceName, NULL);
		printf("%s\n", deviceName);
		_context = clCreateContext(NULL, 1, &current_device, NULL, NULL, &err);
		if (err < 0) {
			perror("create context error\n");
			return false;
		}
		_cmdque = clCreateCommandQueue(_context, current_device, NULL, &err);
		if (err < 0) {
			perror("create command queue error\n");
			return false;
		}
		return true;
	}
	cl_device_id CL_Base::switch_device(int offset) {
		current_device = _devices[offset];
		cl_int err = 0;
		_context = clCreateContext(NULL, 1, &current_device, NULL, NULL, &err);
		if (err < 0) {
			perror("create context error\n");
		}
		_cmdque = clCreateCommandQueue(_context, current_device, NULL, &err);
		if (err < 0) {
			perror("create command queue error\n");
		}
		return current_device;
	}
	bool CL_Base::create_program(const std::string& file) {
		std::fstream kernelfile(file.c_str(), std::fstream::in | std::fstream::out);
		std::string content((std::istreambuf_iterator<char>(kernelfile)),
			std::istreambuf_iterator<char>());
		size_t program_size = content.size();
		const char * kernel = (char*)malloc(content.size());
		kernel = content.c_str();
		cl_int err = 0;
		_program = clCreateProgramWithSource(_context, 1, (const char**)&kernel,
			&program_size, &err);
		if (err < 0) {
			perror("create program error");
			return false;
		}
		size_t log_size = 0;
		char* program_log;
		err = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
		if (err < 0) {
			/* Find size of log and print to std output */
			clGetProgramBuildInfo(_program, current_device, CL_PROGRAM_BUILD_LOG,
				0, NULL, &log_size);
			program_log = (char*)malloc(log_size + 1);
			program_log[log_size] = '\0';
			clGetProgramBuildInfo(_program, current_device, CL_PROGRAM_BUILD_LOG,
				log_size + 1, program_log, NULL);
			printf("%s\n", program_log);
			free(program_log);
			exit(1);
		}
		clGetProgramBuildInfo(_program, current_device, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(_program, current_device, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		return true;
	}
}