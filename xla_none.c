/*
 * xla_none.c
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2014, D. R. Commander.
 * Copyright (C) 2015-2016, 2018, Matthieu Darbois.
 *
 * This file contains stubs for when there is no XLA support available.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"

GLOBAL(void)
jpeg_fdct_xla(FAST_FLOAT *data)
{
	return 0;
}
