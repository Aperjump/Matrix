#define TS 32
#define WPT 8
#define RTS TS/WPT
__kernel void padding(const int M, const int N,
					  const int M_Ex, const int N_Ex,
					  const __global double* input, 
					  __global double* output) {
	const int tx = get_group_id(0) * TS + get_local_id(0);
	const int ty = get_group_id(1) * TS + get_local_id(1);
	if (tx < M_Ex && ty < N_Ex) {
		double value;
		if (tx < M && ty < N) {
			value = input[ty * M + tx];
		} else {
			value = 0;
		}
		output[ty * M_Ex + tx] = value;
	}
}
__kernel void return_padding(const int M_Ex, const int N_Ex,
							 const int M, const int N,
							 const __global double* input,
							 __global double* output) {
	const int tx = get_group_id(0) * TS + get_local_id(0);
	const int ty = get_group_id(1) * TS + get_local_id(1);
	if (tx < M && ty < N) {
		output[ty * M + tx] = input[ty * M_Ex + tx];
	}
}
__kernel void core(const int M, const int N, const int K,
                      const __global double* A,
                      const __global double* B,
                      __global double* C) {
    const int row = get_local_id(0); 
    const int col = get_local_id(1); 
    const int globalrow = TS*get_group_id(0) + row;
    const int globalcol = TS*get_group_id(1) + col;
 
    __local double innerA[TS][TS];
    __local double innerB[TS][TS];
 
    double acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0;
    }
    
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
        for (int w=0; w<WPT; w++) {
            const int tilrow = TS*t + row;
            const int tilcol = TS*t + col;
            innerA[col + w*RTS][row] = A[(tilcol + w*RTS)*M + globalrow];
            innerB[col + w*RTS][row] = B[(globalcol + w*RTS)*K + tilrow];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += innerA[k][row] * innerB[col + w*RTS][k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w=0; w<WPT; w++) {
        C[(globalcol + w*RTS)*M + globalrow] = acc[w];
    }
}
__kernel void add(const int M, const int N, 
					const __global double* A,
					const __global double* B,
					__global double* output)
{
	const int globalrow = get_global_id(0);
	const int globalcol = get_global_id(1);
	double tmp = 0.0;
	tmp = A[globalcol * M + globalrow] + B[globalcol * M + globalrow];
	output[globalcol * M + globalrow] = tmp;
}