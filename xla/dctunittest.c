/*
 * dctunittest.c
 *
 * Since xla only runs dct, unit testing is done by generating random 8x8 blocks,
 * running them on both xla and the original dct, and comparing the outputs.
 */

#define JPEG_INTERNALS
#include "../jinclude.h"
#include "../jpeglib.h"
#include "../jdct.h"
#include <time.h>

#define abs(x) ((x) > 0? (x): -(x))
#define max(x, y) (x > y? x: y)

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCTs.
#endif

static boolean equivalent(FAST_FLOAT *data1, FAST_FLOAT *data2) {
	int i, j;
	for(i = 0; i < DCTSIZE; i++, data1 += DCTSIZE, data2 += DCTSIZE)
		for(j = 0; j < DCTSIZE; j++)
			if(abs(data1[j] - data2[j]) > abs(max(data1[j], data2[j])) * (FAST_FLOAT) REL_ERROR_MARGIN)
				return 0;
	return 1;
}

static boolean test_block(FAST_FLOAT *data) {
	FAST_FLOAT normal_dct[DCTSIZE * DCTSIZE];
	FAST_FLOAT xla_dct[DCTSIZE * DCTSIZE];
random 
	memcpy(normal_dct, data, sizeof normal_dct);
	memcpy(xla_dct, data, sizeof xla_dct);

	jpeg_fdct_float(normal_dct);
	jpeg_fdct_xla(xla_dct);

	return equivalent(normal_dct, xla_dct);
}

/*
 * Create an 8x8 matrix with random values.
 */
static void fill_block(FAST_FLOAT *data) {
	int i, j;
	for(i = 0; i < DCTSIZE; i++, data += DCTSIZE)
		for(j = 0; j < DCTSIZE; j++) {
			data[j] = (FAST_FLOAT) rand() / RAND_MAX * 200;
		}
}

int main(int argc, char *argv[])
{
	FAST_FLOAT block[DCTSIZE * DCTSIZE];
	int i;
	srand((unsigned)time(NULL));

	for(int i = 0; i < TEST_COUNT; i++) {
		fill_block(block);
		if(test_block(block)) {
			fprintf(stderr, "Test succeeded.\n");
		} else {
			fprintf(stderr, "Test failed: xla's dct produced a different result than original dct\n");
			exit(EXIT_FAILURE);
		}
	}
	fprintf(stderr, "All tests succeeded.\n");
	exit(EXIT_SUCCESS);
}

