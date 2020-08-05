/*
 * jfdctxla.c
 *
 * Copyright 2020 Google LLC
 *
 * Stub forward DCT (Discrete Cosine Transform) for XLA.
 *
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"
#include <tensorflow/c/c_api.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <Inference.h>
#include <string.h>

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCTs.
#endif

Inference* inf;
TF_Tensor* in;
FAST_FLOAT* in_data;
int byte_size;
int64_t dims[] = {0, DCTSIZE, DCTSIZE};

GLOBAL(void)
initialize_tf_session(int saiz)
{
  byte_size = saiz * DCTSIZE * DCTSIZE * sizeof(FAST_FLOAT);
  dims[0] = saiz;
  inf = newInference(PB_BINARY_PATH, "x", "y");
  in = TF_AllocateTensor(TF_FLOAT, dims, 3, byte_size);
  in_data = (FAST_FLOAT*)(TF_TensorData(in));
}

GLOBAL(void)
destroy_tf_session()
{
  destroy(inf);
}

GLOBAL(void)
jpeg_fdct_xla(FAST_FLOAT *data)
{
  memcpy(in_data, data, byte_size);
  TF_Tensor* out = runGraph(inf, in);
  FAST_FLOAT* out_data = (FAST_FLOAT*)(TF_TensorData(out));
  memcpy(data, out_data, byte_size);
}

GLOBAL(void)
jpeg_fdct_xla_block(FAST_FLOAT data[][DCTSIZE * DCTSIZE])
{
  memcpy(in_data, data, byte_size);
  TF_Tensor* out = runGraph(inf, in);
  FAST_FLOAT* out_data = (FAST_FLOAT*)(TF_TensorData(out));
  memcpy(data, out_data, byte_size);
}

