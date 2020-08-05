/*
 * dctunittest.c
 *
 * Copyright 2020 Google LLC
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

int block_count;
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

typedef void (*forward_DCT_method_ptr) (DCTELEM *data);
typedef struct {
  double elapsed_time;
} run_info_t;

static void print_info(run_info_t *info) {
  fprintf(stdout, "%lf\n", info->elapsed_time);
}

static void print_block(FAST_FLOAT *data) {
  int i, j;
  for(i = 0; i < DCTSIZE; i++, data += DCTSIZE) {
    for(j = 0; j < DCTSIZE; j++) {
      fprintf(stderr, "%lf ", data[j]);
    }
    fprintf(stderr, "\n");
  }
}

/*
 * We determine two 8x8 blocks to be equivalent if their difference is below a
 * certain relative error. We introduce this maximum allowed relative error as a
 * flag that can be defined at build time.
 */
static boolean are_equivalent(FAST_FLOAT* data1, FAST_FLOAT* data2) {
  int i, j;
  for(i = 0; i < DCTSIZE; i++, data1 += DCTSIZE, data2 += DCTSIZE)
    for(j = 0; j < DCTSIZE; j++)
      if(abs(data1[j] - data2[j]) > abs(max(data1[j], data2[j])) * (FAST_FLOAT) REL_ERROR_MARGIN)
        return 0;
  return 1;
}

static float get_correctness_percentage(FAST_FLOAT original[][DCTSIZE * DCTSIZE], FAST_FLOAT normal_dct[][DCTSIZE * DCTSIZE], FAST_FLOAT xla_dct[][DCTSIZE * DCTSIZE]) {
  int correct_count, u, i, j;
  FAST_FLOAT *data1, *data2;
  correct_count = 0;
  for(u = 0; u < block_count; u++) {
    data1 = normal_dct[u];
    data2 = xla_dct[u];
    if(are_equivalent(data1, data2)) {
      correct_count++;
    }
  }
  return (float) correct_count / block_count;
}

/*
 * Create an 8x8 matrix with random values.
 */
static void fill_block(FAST_FLOAT *data) {
  int i, j;
  for(i = 0; i < DCTSIZE; i++, data += DCTSIZE)
    for(j = 0; j < DCTSIZE; j++) {
      data[j] = (FAST_FLOAT) rand() / RAND_MAX * 255 - 128;
    }
}

int main(int argc, char *argv[])
{
  if(argc < 2) {
    block_count = 1;
  } else {
    block_count = atoi(argv[1]);
  }

  FAST_FLOAT blocks[block_count][DCTSIZE * DCTSIZE];
  FAST_FLOAT normal_dct[block_count][DCTSIZE * DCTSIZE];
  FAST_FLOAT xla_dct[block_count][DCTSIZE * DCTSIZE];
  int i;
  srand(RAND_SEED);

  for(int i = 0; i < block_count; i++) {
    fill_block(blocks[i]);
  }
  memcpy(normal_dct, blocks, sizeof blocks);
  memcpy(xla_dct, blocks, sizeof blocks);

  // Running normal dct
  run_info_t normal_dct_info, xla_dct_info;
  {
    clock_t begin = clock();
    for(i = 0; i < block_count; i++) {
      jpeg_fdct_float(normal_dct[i]);
    }
    normal_dct_info.elapsed_time = (double)(clock() - begin) / CLOCKS_PER_SEC;
    // Scaling normal dct. Not included in time measurement because it's a
    // separate process from the dct.
    for(int u = 0; u < block_count; u++) {
      int i, j;
      FAST_FLOAT *data = normal_dct[u];
      for(i = 0; i < DCTSIZE; i++, data += DCTSIZE)
        for(j = 0; j < DCTSIZE; j++) {
          data[j] /= flow_coeff[i * DCTSIZE + j];
        }
    }
  }

  // Running xla dct
  {
    initialize_tf_session(block_count);
    clock_t begin = clock();
    jpeg_fdct_xla(&xla_dct[0][0]);
    xla_dct_info.elapsed_time = (double)(clock() - begin) / CLOCKS_PER_SEC;
  }

  float correctness = get_correctness_percentage(blocks, normal_dct, xla_dct);
  fprintf(stderr, "Successfully ran the XLA DCT on %d random 8x8 blocks.\n", block_count);
  fprintf(stderr, "Correctness: %.2lf\%\n", correctness * 100);
  fprintf(stderr, "\nRuntime info of normal dct:\n");
  print_info(&normal_dct_info);
  fprintf(stderr, "\nRuntime info of xla dct:\n");
  print_info(&xla_dct_info);
  fprintf(stderr, "\n");

  exit(EXIT_SUCCESS);
}

