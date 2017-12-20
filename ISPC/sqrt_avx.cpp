#include <stdio.h>
#include "immintrin.h"

const float guess = 1;

const __m256 accuracy = _mm256_set1_ps(1e-4);
const __m256 pointFive = _mm256_set1_ps(0.5f); // set 0.5f for computation
const __m256 zero = _mm256_set1_ps(0.f); // set 0.f for computation
const __m256 full_bits = _mm256_set1_ps(-1.f); // full bits to test 0

void sqrt_avx( float* vin, float* vout,int N)
{
	const int width = 8;
	for (int i = 0; i < N; i += width)
	{
		__m256 numInput = _mm256_load_ps(vin + i); // load input
		__m256 numPrev = _mm256_set1_ps(guess); // set the initial guess
		__m256 numCurrent;
		int flag = 0;

		while (!flag)
		{
			// numCurrent = (numPrev + numInput / numPrev) * 0.5f;
			numCurrent = _mm256_div_ps(numInput, numPrev);
			numCurrent = _mm256_add_ps(numPrev, numCurrent);
			numCurrent = _mm256_mul_ps(numCurrent, pointFive);
						
			// get the positive diff
			__m256 diff = _mm256_sub_ps(numCurrent, numPrev);
			__m256 diff_negative = _mm256_sub_ps(zero, diff);
			diff = _mm256_max_ps(diff, diff_negative);
			
			// if diff is greater than accuracy, set to 0x80000000
			// otherwise set to 0 (ordered, non_signal)
			// ps: although the document said when greater than
			// vout would be set to 0xffffffff
			// but in reality it is 0x80000000
			__m256 comp = _mm256_cmp_ps(diff, accuracy, _CMP_GT_OQ);

			// calculate the and of these two inputs and test zero
			// if all zero, set flag to 1, means accuracy is enough
			flag = _mm256_testz_ps(comp, full_bits);

			// put current value as previous and continue loop
			numPrev = numCurrent;
		}

		_mm256_store_ps(vout + i, numCurrent);
	}
}