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
 * Stub forward DCT (Discrete Cosine Transform) for XLA.
 *
 */

#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"               /* Private declarations for DCT subsystem */

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

