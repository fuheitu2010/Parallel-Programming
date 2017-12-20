
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"time.h"
#include"stdlib.h"
#include<cublas.h>
#include<cublas_v2.h>

#include<iomanip>
#include<iostream>
using namespace std;

#define NUM_THREADS 256  

float *a, *b, *c, *d;
float *ag, *bg, *cg, *dg;


clock_t start, finish,start1,finish1;

double duration,duration1;

int row1, column1, row2, column2;

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);

	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");

		return false;
	}

	int i;

	for (i = 0; i < count; i++)
	{

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
	//	printDeviceProp(prop);

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}
	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

void matrix_generate(int i,int j,int i1,int j1)
{
	srand(time(NULL));
	for (int ii = 0; ii < i; ii++)
	{
		for (int jj = 0; jj < j; jj++)
		{
			a[ii*i + jj] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
				
		}
	}

//	srand(time(NULL));
	for (int ii = 0; ii < i1; ii++)
	{
		for (int jj = 0; jj < j1; jj++)
		{
			b[ii*i + jj] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
		}
	}

}

void print_matrix(int row,int column)
{
	cout << "The matrix A is:" << endl;
	int i, j;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < column; j++)
			cout << setw(10) << a[i*row+j];                //Print the elements of the matrix
		cout << endl;
	}
	cout << endl;

	cout << "The matrix B is:" << endl;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < column; j++)
			cout << setw(10) << b[i*row + j];                //Print the elements of the matrix
		cout << endl;
	}
	cout << endl;

	cout << "The result CPU is:" << endl;
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < column; j++)
			cout << setw(10)<<c[i*row + j];                //Print the elements of the matrix
		cout << endl;
	}
	    cout << endl;

}

void printgpu(int row, int column)
{
	cout << "The result GPU is:" << endl;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
			cout << setw(10) << d[i*row + j];                //Print the elements of the matrix
		cout << endl;
	}
	cout << endl;
}

void matrixmul_cpu(int ii, int jj, int ii1, int jj1)
{

	int i, j, q;
		for (i = 0; i < ii; i++)
		{
			for (j = 0; j < jj1; j++)
			{
				c[i*jj+j] = 0;
				for (q = 0; q < ii1; q++)
				{
					c[i*jj + j] = c[i*jj + j] + a[i*jj+q] * b[j+q*ii1];
				}
			}
		}
}



__global__ static void CUDAshare(const float *aa, const float *bb,  float *cc, int n)
{
	extern __shared__ float data[];
	const int tid = threadIdx.x;
	const int row= blockIdx.x;
	int i, j;

	for (i = tid; i< n; i+= blockDim.x) 
	{
	//	printf("%f\n", data[i]);
		data[i] = aa[row*n + i];
		data[i] =5;
		//printf("%f\n", aa[row*n + i]);
		//printf("%f\n",data[i]);
	}



	__syncthreads();

	for (j = tid; j < n; j+= blockDim.x) {
		float t = 0;
		float y= 0;
		for (i  = 0; i < n; i++) {
			float r;

		//	printf("%f\n", bb[i * n + j]);
		//	printf("%f\n", data[i]);

			y-= data[i] * bb[i * n  + j];
			r= t - y;
			y= (r - t) + y;
			t = r;
		}
		cc[row*n + j] = t;
	}
}


__global__ void matrixgpu(float *aa, float *bb, float *cc, int n)
{
	
	
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	int row = index / n;
	int column = index%n;
	if (row < n && column < n)
	{
		float t = 0;

		for (int ii = 0; ii < n; ii=ii+5)
		{
			t = t + aa[row*n + ii] * bb[ii*n + column];
			t = t + aa[row*n + ii + 1] * bb[(ii + 1)*n + column];
			t = t + aa[row*n + ii + 2] * bb[(ii + 2)*n + column];
			t = t + aa[row*n + ii + 3] * bb[(ii + 3)*n + column];
			t = t + aa[row*n + ii + 4] * bb[(ii + 4)*n + column];

		}
		cc[row*n + column] = t;
	}

}



int main()
{
	int block;
	cout<<"Input the dimension matrix"<<endl;
	cin >> row1;

	cout << "The dimension of matrixes: " << row1 << endl;

	if (InitCUDA() == false)
	{
		return 0;
	}

	a = (float*)malloc(sizeof(float)* row1 * row1);
	b = (float*)malloc(sizeof(float)* row1 * row1);
	c = (float*)malloc(sizeof(float)* row1 * row1);
	d = (float*)malloc(sizeof(float)* row1 * row1);

	cudaMalloc((void**)&ag, sizeof(float)*row1*row1);
	cudaMalloc((void**)&bg, sizeof(float)*row1*row1);
	cudaMalloc((void**)&cg, sizeof(float)*row1*row1);
	
	matrix_generate(row1,row1,row1,row1 );

	
	  cudaMemcpy(ag, a, sizeof(float)*row1*row1, cudaMemcpyHostToDevice);
	  cudaMemcpy(bg, b, sizeof(float)*row1*row1, cudaMemcpyHostToDevice);

	start = clock();

	matrixmul_cpu(row1, row1, row1, row1);
	finish = clock();
	duration = (double)(finish - start) /( CLOCKS_PER_SEC/1000);

	cout << "The CPU time is " << duration<<" ms"<<endl;

	block = 1+row1*row1 / NUM_THREADS;

	start1 = clock();
	
	//CUDAshare <<<block, NUM_THREADS >>>(ag,bg, cg, row1);
	matrixgpu << <block, NUM_THREADS >> >(ag, bg, cg, row1);

	cudaThreadSynchronize();

	finish1 = clock();
	
	cudaMemcpy(d, cg, sizeof(float)* row1*row1, cudaMemcpyDeviceToHost);

	duration1 = (double)(finish1 - start1) / (CLOCKS_PER_SEC / 1000);

	cout << "The GPU time is " << duration1 << " ms" << endl;


	print_matrix(row1, row1);

	printgpu(row1, row1);
	cudaFree(ag);
	cudaFree(bg);
	cudaFree(cg);

    return 0;
}


