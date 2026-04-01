/* Copyright 2025 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 */

#ifndef _SHA1_H
#define _SHA1_H

/* This is not SHA1. This is BLAKE3 exposed as SHA1 functions due to
 * transitional and historic reasons.
 *
 * TODO: Remove this and use _mesa_blake3_* functions everywhere.
 * All remnants of SHA1 should be removed from Mesa except build_id.
 */

#include "util/mesa-blake3.h"

#define	SHA1_DIGEST_LENGTH		BLAKE3_KEY_LEN
#define	SHA1_DIGEST_STRING_LENGTH	BLAKE3_HEX_LEN

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _SHA1_CTX {
    blake3_hasher hasher;
} SHA1_CTX;

static inline void
SHA1Init(SHA1_CTX *context)
{
    _mesa_blake3_init(&context->hasher);
}

static inline void
SHA1Update(SHA1_CTX *context, const uint8_t *data, size_t len)
{
    _mesa_blake3_update(&context->hasher, data, len);
}

static inline void
SHA1Final(uint8_t digest[SHA1_DIGEST_LENGTH], SHA1_CTX *context)
{
    _mesa_blake3_final(&context->hasher, digest);
}

#ifdef __cplusplus
}
#endif

#endif /* _SHA1_H */
