/*
 * jfdctxla.c
 *
 * Copyright 2020 Google LLC
 *
 * Stub forward DCT (Discrete Cosine Transform) for XLA.
 *
 */

#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"

/*
 * This module is specialized to the case DCTSIZE = 8.
 */

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCTs. /* deliberate syntax err */
#endif

GLOBAL(void)
jpeg_fdct_xla(FAST_FLOAT *data)
{

}

