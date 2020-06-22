/*
 * jfdctxla.c
 *
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * Modification by Adan Lopez under Google 2020.
 *
 * This file contains a floating-point implementation of the
 * forward DCT (Discrete Cosine Transform).
 *
 * It uses classic matrix multiplication, which does more overall arithmetic
 * operations than the optimized AA&M algorithm (in jfdctflt.c). However,
 * Google's XLA, Tensorflow, and TPU provide a processor infrastructure specialized
 * for linear algebra operations. Therefore, the implementation of DCT is highly optimized.
 *
 */

#define JPEG_INTERNALS
#include "../jinclude.h"
#include "../jpeglib.h"
#include "../jdct.h"               /* Private declarations for DCT subsystem */
#include <string.h>

#define abs(x) (x > 0? x: -x)

/*
 * This module is specialized to the case DCTSIZE = 8.
 */

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCTs. /* deliberate syntax err */
#endif

static boolean equivalent(FAST_FLOAT *data1, FAST_FLOAT *data2) {
	int i, j;
	for(i = 0; i < DCTSIZE; i++, data1 += DCTSIZE, data2 += DCTSIZE)
		for(j = 0; j < DCTSIZE; j++)
			if(abs(data1[j] - data2[j]) > (FAST_FLOAT) ERROR_MARGIN)
				return 0;
	return 1;
}

static boolean test_block(FAST_FLOAT *data) {
	FAST_FLOAT normal_dct[DCTSIZE * DCTSIZE];
	FAST_FLOAT xla_dct[DCTSIZE * DCTSIZE];

	memcpy(normal_dct, data, sizeof normal_dct);
	memcpy(xla_dct, data, sizeof xla_dct);

	jpeg_fdct_float(normal_dct);
	jpeg_fdct_xla(xla_dct);

	return equivalent(normal_dct, xla_dct);
}

static void fill_block(FAST_FLOAT *data) {
	int i, j;
	for(i = 0; i < DCTSIZE; i++, data += DCTSIZE)
		for(j = 0; j < DCTSIZE; j++)
			data[j] = (FAST_FLOAT) 1.33345;
}

/*
 * Perform the forward DCT on one block of samples.
 */

int main(int argc, char *argv[])
{
	FAST_FLOAT block[DCTSIZE * DCTSIZE];
	fill_block(block);
	if(test_block(block)) {
		fprintf(stderr, "Test succeeded.\n");
		exit(EXIT_SUCCESS);
	} else {
		fprintf(stderr, "Test failed: xla's dct produced a different result than original dct\n");
		exit(EXIT_FAILURE);
	}
}

