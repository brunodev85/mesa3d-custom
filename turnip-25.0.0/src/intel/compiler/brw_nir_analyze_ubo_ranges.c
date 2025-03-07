/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "brw_eu.h"
#include "brw_nir.h"
#include "compiler/nir/nir.h"
#include "util/u_dynarray.h"

/**
 * \file brw_nir_analyze_ubo_ranges.c
 *
 * This pass decides which portions of UBOs to upload as push constants,
 * so shaders can access them as part of the thread payload, rather than
 * having to issue expensive memory reads to pull the data.
 *
 * The 3DSTATE_CONSTANT_* mechanism can push data from up to 4 different
 * buffers, in GRF sized units.  This was always 256 bits (32 bytes).
 * Starting with Xe2, it is 512 bits (64 bytes).
 *
 * To do this, we examine NIR load_ubo intrinsics, recording the number of
 * loads at each offset.  We track offsets at a sizeof(GRF) granularity, so even
 * fields with a bit of padding between them tend to fall into contiguous
 * ranges.  We build a list of these ranges, tracking their "cost" (number
 * of registers required) and "benefit" (number of pull loads eliminated
 * by pushing the range).  We then sort the list to obtain the four best
 * ranges (most benefit for the least cost).
 */

struct ubo_range_entry
{
   struct brw_ubo_range range;
   int benefit;
};

static int
score(const struct ubo_range_entry *entry)
{
   return 2 * entry->benefit - entry->range.length;
}

/**
 * Compares score for two UBO range entries.
 *
 * For a descending qsort().
 */
static int
cmp_ubo_range_entry(const void *va, const void *vb)
{
   const struct ubo_range_entry *a = va;
   const struct ubo_range_entry *b = vb;

   /* Rank based on scores, descending order */
   int delta = score(b) - score(a);

   /* Then use the UBO block index as a tie-breaker, descending order */
   if (delta == 0)
      delta = b->range.block - a->range.block;

   /* Finally use the start offset as a second tie-breaker, ascending order */
   if (delta == 0)
      delta = a->range.start - b->range.start;

   return delta;
}

struct ubo_block_info
{
   /* Each bit in the offsets bitfield represents a 32-byte section of data.
    * If it's set to one, there is interesting UBO data at that offset.  If
    * not, there's a "hole" - padding between data - or just nothing at all.
    */
   uint64_t offsets;
   uint8_t uses[64];
};

struct ubo_analysis_state
{
   struct hash_table *blocks;
   const struct intel_device_info *devinfo;
};

static struct ubo_block_info *
get_block_info(struct ubo_analysis_state *state, int block)
{
   uint32_t hash = block + 1;
   void *key = (void *) (uintptr_t) hash;

   struct hash_entry *entry =
      _mesa_hash_table_search_pre_hashed(state->blocks, hash, key);

   if (entry)
      return (struct ubo_block_info *) entry->data;

   struct ubo_block_info *info =
      rzalloc(state->blocks, struct ubo_block_info);
   _mesa_hash_table_insert_pre_hashed(state->blocks, hash, key, info);

   return info;
}

static void
analyze_ubos_block(struct ubo_analysis_state *state, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic != nir_intrinsic_load_ubo)
         continue;

      if (brw_nir_ubo_surface_index_is_pushable(intrin->src[0]) &&
          nir_src_is_const(intrin->src[1])) {
         const int block = brw_nir_ubo_surface_index_get_push_block(intrin->src[0]);
         const unsigned byte_offset = nir_src_as_uint(intrin->src[1]);
         const unsigned sizeof_GRF = REG_SIZE * reg_unit(state->devinfo);
         const int offset = byte_offset / sizeof_GRF;

         /* Avoid shifting by larger than the width of our bitfield, as this
          * is undefined in C.  Even if we require multiple bits to represent
          * the entire value, it's OK to record a partial value - the backend
          * is capable of falling back to pull loads for later components of
          * vectors, as it has to shrink ranges for other reasons anyway.
          */
         if (offset >= 64)
            continue;

         /* The value might span multiple sizeof(GRF) chunks. */
         const unsigned num_components =
            nir_def_last_component_read(&intrin->def) + 1;
         const int bytes = num_components * (intrin->def.bit_size / 8);
         const int start = ROUND_DOWN_TO(byte_offset, sizeof_GRF);
         const int end = ALIGN(byte_offset + bytes, sizeof_GRF);
         const int chunks = (end - start) / sizeof_GRF;

         /* TODO: should we count uses in loops as higher benefit? */

         struct ubo_block_info *info = get_block_info(state, block);
         info->offsets |= ((1ull << chunks) - 1) << offset;
         info->uses[offset]++;
      }
   }
}

static void
print_ubo_entry(FILE *file,
                const struct ubo_range_entry *entry,
                struct ubo_analysis_state *state)
{
   struct ubo_block_info *info = get_block_info(state, entry->range.block);

   fprintf(file,
           "block %2d, start %2d, length %2d, bits = %"PRIx64", "
           "benefit %2d, cost %2d, score = %2d\n",
           entry->range.block, entry->range.start, entry->range.length,
           info->offsets, entry->benefit, entry->range.length, score(entry));
}

void
brw_nir_analyze_ubo_ranges(const struct brw_compiler *compiler,
                           nir_shader *nir,
                           struct brw_ubo_range out_ranges[4])
{
   void *mem_ctx = ralloc_context(NULL);

   struct ubo_analysis_state state = {
      .blocks =
         _mesa_hash_table_create(mem_ctx, NULL, _mesa_key_pointer_equal),
      .devinfo = compiler->devinfo,
   };

   /* Walk the IR, recording how many times each UBO block/offset is used. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         analyze_ubos_block(&state, block);
      }
   }

   /* Find ranges: a block, starting register-size aligned byte offset, and
    * length.
    */
   struct util_dynarray ranges;
   util_dynarray_init(&ranges, mem_ctx);

   hash_table_foreach(state.blocks, entry) {
      const int b = entry->hash - 1;
      const struct ubo_block_info *info = entry->data;
      uint64_t offsets = info->offsets;

      /* Walk through the offsets bitfield, finding contiguous regions of
       * set bits:
       *
       *   0000000001111111111111000000000000111111111111110000000011111100
       *            ^^^^^^^^^^^^^            ^^^^^^^^^^^^^^        ^^^^^^
       *
       * Each of these will become a UBO range.
       */
      while (offsets != 0) {
         /* Find the first 1 in the offsets bitfield.  This represents the
          * start of a range of interesting UBO data.  Make it zero-indexed.
          */
         int first_bit = ffsll(offsets) - 1;

         /* Find the first 0 bit in offsets beyond first_bit.  To find the
          * first zero bit, we find the first 1 bit in the complement.  In
          * order to ignore bits before first_bit, we mask off those bits.
          */
         int first_hole = ffsll(~offsets & ~((1ull << first_bit) - 1)) - 1;

         if (first_hole == -1) {
            /* If we didn't find a hole, then set it to the end of the
             * bitfield.  There are no more ranges to process.
             */
            first_hole = 64;
            offsets = 0;
         } else {
            /* We've processed all bits before first_hole.  Mask them off. */
            offsets &= ~((1ull << first_hole) - 1);
         }

         struct ubo_range_entry *entry =
            util_dynarray_grow(&ranges, struct ubo_range_entry, 1);

         entry->range.block = b;
         entry->range.start = first_bit;
         /* first_hole is one beyond the end, so we don't need to add 1 */
         entry->range.length = first_hole - first_bit;
         entry->benefit = 0;

         for (int i = 0; i < entry->range.length; i++)
            entry->benefit += info->uses[first_bit + i];
      }
   }

   int nr_entries = ranges.size / sizeof(struct ubo_range_entry);

   if (0) {
      util_dynarray_foreach(&ranges, struct ubo_range_entry, entry) {
         print_ubo_entry(stderr, entry, &state);
      }
   }

   /* TODO: Consider combining ranges.
    *
    * We can only push 4 ranges via 3DSTATE_CONSTANT_XS.  If there are
    * more ranges, and two are close by with only a small hole, it may be
    * worth combining them.  The holes will waste register space, but the
    * benefit of removing pulls may outweigh that cost.
    */

   /* Sort the list so the most beneficial ranges are at the front. */
   if (nr_entries > 0) {
      qsort(ranges.data, nr_entries, sizeof(struct ubo_range_entry),
            cmp_ubo_range_entry);
   }

   struct ubo_range_entry *entries = ranges.data;

   /* Return the top 4, limited to the maximum number of push registers.
    *
    * The Vulkan driver sets up additional non-UBO push constants, so it may
    * need to shrink these ranges further (see anv_nir_compute_push_layout.c).
    * The OpenGL driver treats legacy uniforms as a UBO, so this is enough.
    *
    * To limit further, simply drop the tail of the list, as that's the least
    * valuable portion.
    */
   const int max_ubos = 4;
   nr_entries = MIN2(nr_entries, max_ubos);

   const unsigned max_push_regs = 64 / reg_unit(compiler->devinfo);
   unsigned total_push_regs = 0;

   for (unsigned i = 0; i < nr_entries; i++) {
      if (total_push_regs + entries[i].range.length > max_push_regs)
         entries[i].range.length = max_push_regs - total_push_regs;
      total_push_regs += entries[i].range.length;
   }

   for (int i = 0; i < nr_entries; i++) {
      out_ranges[i] = entries[i].range;

      /* To this point, various values have been tracked in terms of the real
       * hardware register sizes.  However, the rest of the compiler expects
       * values in terms of pre-Xe2 256-bit registers. Scale start and length
       * to account for this.
       */
      out_ranges[i].start *= reg_unit(compiler->devinfo);
      out_ranges[i].length *= reg_unit(compiler->devinfo);
   }
   for (int i = nr_entries; i < 4; i++) {
      out_ranges[i].block = 0;
      out_ranges[i].start = 0;
      out_ranges[i].length = 0;
   }

   ralloc_free(ranges.mem_ctx);
}
