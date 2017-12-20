#include <stdio.h>
#include <stdlib.h>
//#include <ctime>
#include "timing.h"
#include "immintrin.h"
// Include the header file that the ispc compiler generates
#include "sqrt_ispc.h"
using namespace ispc;

void sqrt_seq(float vin[],float vout[],int num_element);
extern void sqrt_avx( float* vin, float* vout,int N);

int main(int argc, char* argv[]) 
{
//	const int N=15000000;
// float vin[N],vout[N];
	int N=atoi(argv[1]);
	int num_task =atoi(argv[2]);
	float *vin=new float [N];
	float *vout=new float [N];
	for (int i=0 ; i<N ; i++)
	{
		vin[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/5));
	}

	printf("\nThis is sequential version vout\n");
	reset_and_start_timer();
	sqrt_seq(vin,vout,N);
	double onecycle= get_elapsed_mcycles();
	double end1=get_elapsed_msec();
	printf("Telapsed processor cycles serial run :\t%.3f million cycles\n", onecycle);
	printf("Sqeuential Sqrt Run time: %.3fms\n\n",end1);


	for (int i=0 ; i<N ; i++) 
		vout[i]=0;
	printf("This is single cpu core(no tasks launched) vout\n");
	reset_and_start_timer();
	sqrtn(vin, vout, N);
	double twocycle = get_elapsed_mcycles();
	double end2=get_elapsed_msec();
	printf("Time of single core run :\t%.3f million cycles\n", twocycle);
	printf("Single Core(no tasks launched) Run time: %.3fms\n",end2);
	printf("Speedup : %.3f\n\n",end1/end2);
	
	for (int i=0 ; i<N ; i++) 
		vout[i]=0;
	printf("This is Multiple Cores(with %d tasks)vout\n",num_task);
	reset_and_start_timer();
	ispc_task(vin, vout, N,num_task);
	double threecycle= get_elapsed_mcycles();
	double end3=get_elapsed_msec();
	printf("Time of multiple core run :\t%.3f million cycles\n",  threecycle);
	printf("Multiple Cores(with %d tasks) Run time: %.3fms\n",num_task,end3);
	printf("Speedup : %.3f\n\n",end1/end3);

	for (int i=0 ; i<N ; i++) 
		vout[i]=0;
	printf("This is the AVX intrinsics...\n");
	reset_and_start_timer();
	sqrt_avx(vin, vout, N);
	double fourcycle= get_elapsed_mcycles();
	double end4=get_elapsed_msec();
	printf("Time of 8-wide AVX run :\t%.3f million cycles\n", fourcycle);
	printf("AVX intrinsic with %d tasks) Run time: %.3fms\n ",num_task,end4);
	printf("Speedup : %.3f\n",end1/end4);

	delete[] vin;
	delete[] vout;
	return 0;
}

void sqrt_seq(float vin[],float vout[],int num_element)
{
	for(int i=0;i<num_element;i++)
	{
		float f=1;
	float dev=((f*f - vin[i])>0?(f*f - vin[i]):(vin[i]-f*f));
	while(dev >= 0.0001 )
    {
       f = ((vin[i]/f) + f) / 2;
       dev=((f*f - vin[i])>0?(f*f - vin[i]):(vin[i]-f*f));
    }
    vout[i]=f;
}
}