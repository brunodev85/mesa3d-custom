/*
 * Copyright (C) 2020-2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "pan_nir.h"

/*
 * If the shader packs multiple varyings into the same location with different
 * location_frac, we'll need to lower to a single varying store that collects
 * all of the channels together. This is because the varying instruction on
 * Midgard and Bifrost is slot-based, writing out an entire vec4 slot at a time.
 *
 * NOTE: this expects all stores to be outside of control flow, and with
 * constant offsets. It should be run after nir_lower_io_vars_to_temporaries.
 */
static bool
lower_store_component(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output &&
       intr->intrinsic != nir_intrinsic_store_per_view_output)
      return false;

   struct hash_table_u64 *slots = data;
   unsigned component = nir_intrinsic_component(intr);
   nir_src *slot_src = nir_get_io_offset_src(intr);
   uint64_t slot = nir_src_as_uint(*slot_src) + nir_intrinsic_base(intr);

   if (intr->intrinsic == nir_intrinsic_store_per_view_output) {
      uint64_t view_index = nir_src_as_uint(intr->src[1]);
      slot |= view_index << 32;
   }

   nir_intrinsic_instr *prev = _mesa_hash_table_u64_search(slots, slot);
   unsigned mask = (prev ? nir_intrinsic_write_mask(prev) : 0);

   nir_def *value = intr->src[0].ssa;
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *undef = nir_undef(b, 1, value->bit_size);
   nir_def *channels[4] = {undef, undef, undef, undef};

   /* Copy old */
   u_foreach_bit(i, mask) {
      assert(prev != NULL);
      nir_def *prev_ssa = prev->src[0].ssa;
      channels[i] = nir_channel(b, prev_ssa, i);
   }

   /* Copy new */
   unsigned new_mask = nir_intrinsic_write_mask(intr);
   mask |= (new_mask << component);

   u_foreach_bit(i, new_mask) {
      assert(component + i < 4);
      channels[component + i] = nir_channel(b, value, i);
   }

   intr->num_components = util_last_bit(mask);
   nir_src_rewrite(&intr->src[0], nir_vec(b, channels, intr->num_components));

   nir_intrinsic_set_component(intr, 0);
   nir_intrinsic_set_write_mask(intr, mask);

   if (prev) {
      _mesa_hash_table_u64_remove(slots, slot);
      nir_instr_remove(&prev->instr);
   }

   _mesa_hash_table_u64_insert(slots, slot, intr);
   return true;
}

bool
pan_nir_lower_store_component(nir_shader *s)
{
   assert(s->info.stage == MESA_SHADER_VERTEX);

   struct hash_table_u64 *stores = _mesa_hash_table_u64_create(NULL);
   bool progress = nir_shader_intrinsics_pass(
      s, lower_store_component,
      nir_metadata_control_flow, stores);
   _mesa_hash_table_u64_destroy(stores);
   return progress;
}
