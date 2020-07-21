/*
 * xla_none.c
 *
 * Copyright 2020 Google LLC
 *
 * This file contains stubs for when there is no XLA support available.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"

GLOBAL(void)
initialize_tf_session()
{

}

GLOBAL(void)
destroy_tf_session()
{

}

GLOBAL(void)
jpeg_fdct_xla(FAST_FLOAT *data)
{

}
