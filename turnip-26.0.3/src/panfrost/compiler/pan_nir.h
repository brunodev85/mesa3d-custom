/*
 * Copyright (C) 2025 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef __PAN_NIR_H__
#define __PAN_NIR_H__

#include "nir.h"
#include "nir_builder.h"
#include "pan_compiler.h"

struct util_format_description;

static inline nir_def *
pan_nir_tile_rt_sample(nir_builder *b, nir_def *rt, nir_def *sample)
{
   /* y = 255 means "current pixel" */
   return nir_pack_32_4x8_split(b, nir_u2u8(b, sample),
                                   nir_u2u8(b, rt),
                                   nir_imm_intN_t(b, 0, 8),
                                   nir_imm_intN_t(b, 255, 8));
}

static inline nir_def *
pan_nir_tile_location_sample(nir_builder *b, gl_frag_result location,
                             nir_def *sample)
{
   uint8_t rt;
   if (location == FRAG_RESULT_DEPTH) {
      rt = 255;
   } else if (location == FRAG_RESULT_STENCIL) {
      rt = 254;
   } else {
      assert(location >= FRAG_RESULT_DATA0);
      rt = location - FRAG_RESULT_DATA0;
   }

   return pan_nir_tile_rt_sample(b, nir_imm_int(b, rt), sample);
}

static inline nir_def *
pan_nir_tile_default_coverage(nir_builder *b)
{
   return nir_iand_imm(b, nir_load_cumulative_coverage_pan(b), 0x1f);
}

bool pan_nir_lower_store_component(nir_shader *shader);

bool pan_nir_lower_vertex_id(nir_shader *shader);

bool pan_nir_lower_image_ms(nir_shader *shader);

bool pan_nir_lower_frag_coord_zw(nir_shader *shader);
bool pan_nir_lower_noperspective_vs(nir_shader *shader);
bool pan_nir_lower_noperspective_fs(nir_shader *shader);

bool pan_nir_lower_helper_invocation(nir_shader *shader);
bool pan_nir_lower_sample_pos(nir_shader *shader);
bool pan_nir_lower_xfb(nir_shader *nir);

bool pan_nir_lower_image_index(nir_shader *shader,
                               unsigned vs_img_attrib_offset);
bool pan_nir_lower_texel_buffer_fetch_index(nir_shader *shader,
                                            unsigned attrib_offset);

void pan_nir_lower_texture_early(nir_shader *nir, unsigned gpu_id);
void pan_nir_lower_texture_late(nir_shader *nir, unsigned gpu_id);

nir_alu_type
pan_unpacked_type_for_format(const struct util_format_description *desc);

bool pan_nir_lower_framebuffer(nir_shader *shader,
                               const enum pipe_format *rt_fmts,
                               uint8_t raw_fmt_mask,
                               unsigned blend_shader_nr_samples,
                               bool broken_ld_special);

bool pan_nir_lower_fs_outputs(nir_shader *shader, bool skip_atest);

uint32_t pan_nir_collect_noperspective_varyings_fs(nir_shader *s);

/* Specify the mediump lowering behavior for pan_nir_collect_varyings */
enum pan_mediump_vary {
   /* Always assign a 32-bit format to mediump varyings */
   PAN_MEDIUMP_VARY_32BIT,
   /* Assign a 16-bit format to varyings with smooth interpolation, and a
    * 32-bit format to varyings with flat interpolation */
   PAN_MEDIUMP_VARY_SMOOTH_16BIT,
};

void pan_nir_collect_varyings(nir_shader *s, struct pan_shader_info *info,
                              enum pan_mediump_vary mediump);

#endif /* __PAN_NIR_H__ */
