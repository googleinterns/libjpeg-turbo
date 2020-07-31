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
  fprintf(stderr, "Elapsed time: %lf\n", info->elapsed_time);
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
static void assert_equivalent(FAST_FLOAT original[][DCTSIZE * DCTSIZE], FAST_FLOAT normal_dct[][DCTSIZE * DCTSIZE], FAST_FLOAT xla_dct[][DCTSIZE * DCTSIZE]) {
  int u, i, j;
  FAST_FLOAT *data1, *data2;
  for(u = 0; u < BLOCK_COUNT; u++) {
    data1 = normal_dct[u];
    data2 = xla_dct[u];
    for(i = 0; i < DCTSIZE; i++, data1 += DCTSIZE, data2 += DCTSIZE) {
      for(j = 0; j < DCTSIZE; j++) {
        if(abs(data1[j] - data2[j]) > abs(max(data1[j], data2[j])) * (FAST_FLOAT) REL_ERROR_MARGIN) {
          fprintf(stderr, "%d-th block failed\n", u + 1);
          fprintf(stderr, "Normal dct and Xla dct didn't produce the same result for a specific 8x8 block\n");
          fprintf(stderr, "\n8x8 block:\n");
          print_block(original[u]);
          fprintf(stderr, "\nNormal dct produced:\n");
          print_block(normal_dct[u]);
          fprintf(stderr, "\nXla dct produced:\n");
          print_block(xla_dct[u]);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
}

static run_info_t run_dct(forward_DCT_method_ptr do_dct, FAST_FLOAT data[][DCTSIZE * DCTSIZE]) {
  run_info_t info;
  int i;
  clock_t begin = clock();
  for(i = 0; i < BLOCK_COUNT; i++) {
    (*do_dct) (data[i]);
  }
  info.elapsed_time = (double)(clock() - begin) / CLOCKS_PER_SEC;
  return info;
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
  FAST_FLOAT blocks[BLOCK_COUNT][DCTSIZE * DCTSIZE];
  FAST_FLOAT normal_dct[BLOCK_COUNT][DCTSIZE * DCTSIZE];
  FAST_FLOAT xla_dct[BLOCK_COUNT][DCTSIZE * DCTSIZE];
  int i;
  srand(RAND_SEED);

  for(int i = 0; i < BLOCK_COUNT; i++) {
    fill_block(blocks[i]);
  }
  memcpy(normal_dct, blocks, sizeof blocks);
  memcpy(xla_dct, blocks, sizeof blocks);

  // Running normal dct
  run_info_t normal_dct_info = run_dct(jpeg_fdct_float, normal_dct);
  for(int u = 0; u < BLOCK_COUNT; u++) {
    int i, j;
    FAST_FLOAT *data = normal_dct[u];
    for(i = 0; i < DCTSIZE; i++, data += DCTSIZE)
      for(j = 0; j < DCTSIZE; j++) {
        data[j] /= flow_coeff[i * DCTSIZE + j];
      }
    // Coefficients needed for normal (scaled) dct
  }

  // Running xla dct
  initialize_tf_session(); // Necessary for xla dct

  run_info_t xla_dct_info = run_dct(jpeg_fdct_xla, xla_dct);

  assert_equivalent(blocks, normal_dct, xla_dct);
  fprintf(stderr, "All tests succeeded.\n");
  fprintf(stderr, "\nRuntime info of normal dct:\n");
  print_info(&normal_dct_info);
  fprintf(stderr, "\nRuntime info of xla dct:\n");
  print_info(&xla_dct_info);
  fprintf(stderr, "\n");

  exit(EXIT_SUCCESS);
}

