/*
 * jfdctxla.c
 *
 * Modification by Adan Lopez under Google 2020.
 *
 * Stub forward DCT (Discrete Cosine Transform) for XLA.
 *
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"
#include <math.h>

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCTs.
#endif

FAST_FLOAT dct_matrix[DCTSIZE * DCTSIZE] = {
	0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391,
	0.490392640, 0.415734806, 0.277785117, 0.097545161, -0.097545161, -0.277785117, -0.415734806, -0.490392640,
	0.461939766, 0.191341716, -0.191341716, -0.461939766, -0.461939766, -0.191341716, 0.191341716, 0.461939766,
	0.415734806, -0.097545161, -0.490392640, -0.277785117, 0.277785117, 0.490392640, 0.097545161, -0.415734806,
	0.353553391, -0.353553391, -0.353553391, 0.353553391, 0.353553391, -0.353553391, -0.353553391, 0.353553391,
	0.277785117, -0.490392640, 0.097545161, 0.415734806, -0.415734806, -0.097545161, 0.490392640, -0.277785117,
	0.191341716, -0.461939766, 0.461939766, -0.191341716, -0.191341716, 0.461939766, -0.461939766, 0.191341716,
	0.097545161, -0.277785117, 0.415734806, -0.490392640, 0.490392640, -0.415734806, 0.277785117, -0.097545161
};

FAST_FLOAT transposed_dct_matrix[DCTSIZE * DCTSIZE] = {
	0.353553391, 0.490392640, 0.461939766, 0.415734806, 0.353553391, 0.277785117, 0.191341716, 0.097545161,
	0.353553391, 0.415734806, 0.191341716, -0.097545161, -0.353553391, -0.490392640, -0.461939766, -0.277785117,
	0.353553391, 0.277785117, -0.191341716, -0.490392640, -0.353553391, 0.097545161, 0.461939766, 0.415734806,
	0.353553391, 0.097545161, -0.461939766, -0.277785117, 0.353553391, 0.415734806, -0.191341716, -0.490392640,
	0.353553391, -0.097545161, -0.461939766, 0.277785117, 0.353553391, -0.415734806, -0.191341716, 0.490392640,
	0.353553391, -0.277785117, -0.191341716, 0.490392640, -0.353553391, -0.097545161, 0.461939766, -0.415734806,
	0.353553391, -0.415734806, 0.191341716, 0.097545161, -0.353553391, 0.490392640, -0.461939766, 0.277785117,
	0.353553391, -0.490392640, 0.461939766, -0.415734806, 0.353553391, -0.277785117, 0.191341716, -0.097545161
};

FAST_FLOAT flow_coeff[DCTSIZE * DCTSIZE] = {
	8.000000,  11.096321, 10.452571, 9.406975,  8.000002,  6.285559,  4.329570,  2.207195,
	11.096320, 15.391046, 14.498035, 13.047892, 11.096315, 8.718322,  6.005284,  3.061468,
	10.452501, 14.498041, 13.656854, 12.290866, 10.452503, 8.212481,  5.656853,  2.883838,
	9.407003,  13.047894, 12.290842, 11.061483, 9.407002,  7.391037,  5.091034,  2.595389,
	8.000001,  11.096319, 10.452504, 9.407007,  8.000000,  6.285560,  4.329569,  2.207195,
	6.285562,  8.718323,  8.212480,  7.391035,  6.285561,  4.938532,  3.401721,  1.734182,
	4.329566,  6.005280,  5.656854,  5.091043,  4.329567,  3.401721,  2.343146,  1.194525,
	2.207195,  3.061468,  2.883839,  2.595387,  2.207199,  1.734182,  1.194525,  0.608964
};

FAST_FLOAT tmp[DCTSIZE * DCTSIZE];

/*
 * Multiplies 8x8 matrices matrix1 and matrix2, and writes the answer in result,
 */
void matrix_multiply(FAST_FLOAT *matrix1, FAST_FLOAT *matrix2, FAST_FLOAT *result) {
	/* It's possible that result pointer is the same as matrix1 or matrix2. Therefore, writing the
	 * answer directly to result is unsafe. So we store the answer in aux during the multiplication,
	 * and copy it to result in the end.
	 */
	int i, j, k;
	FAST_FLOAT operand1, operand2, *current_cell;

	for(i = 0; i < DCTSIZE; i++) {
		for(j = 0; j < DCTSIZE; j++) {
			current_cell = tmp + i * DCTSIZE + j;
			*current_cell = 0;
			for(k = 0; k < DCTSIZE; k++) {
				operand1 = matrix1[i * DCTSIZE + k];
				operand2 = matrix2[k * DCTSIZE + j];

				*current_cell += operand1 * operand2;
			}
		}
	}
	memcpy(result, tmp, sizeof tmp);
}

GLOBAL(void)
jpeg_fdct_xla(FAST_FLOAT *data)
{
	fprintf(stderr, "In xla dct\n");
	matrix_multiply(dct_matrix, data, data);
	matrix_multiply(data, transposed_dct_matrix, data);
	
	int i;
	for(int i = 0; i < DCTSIZE * DCTSIZE; i++)
		data[i] *= flow_coeff[i];
}

