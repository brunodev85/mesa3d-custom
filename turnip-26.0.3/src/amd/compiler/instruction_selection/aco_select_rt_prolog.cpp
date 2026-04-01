/*
 * Copyright © 2023 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "aco_builder.h"
#include "aco_instruction_selection.h"
#include "aco_interface.h"
#include "aco_ir.h"
#include "aco_nir_call_attribs.h"

#include "ac_descriptors.h"
#include "sid.h"

namespace aco {

void
select_rt_prolog(Program* program, ac_shader_config* config,
                 const struct aco_compiler_options* options, const struct aco_shader_info* info,
                 const struct ac_shader_args* in_args, const struct ac_arg* descriptors,
                 unsigned raygen_param_count, nir_parameter* raygen_params)
{
   init_program(program, compute_cs, info, options, config);
   Block* block = program->create_and_insert_block();
   block->kind = block_kind_top_level;
   program->workgroup_size = info->workgroup_size;
   program->wave_size = info->workgroup_size;
   calc_min_waves(program);
   Builder bld(program, block);
   block->instructions.reserve(32);
   unsigned num_sgprs = in_args->num_sgprs_used;
   unsigned num_vgprs = in_args->num_vgprs_used;

   RegisterDemand limit = get_addr_regs_from_waves(program, program->min_waves);

   struct callee_info raygen_info =
      get_callee_info(program->gfx_level, program->wave_size, rtRaygenABI, raygen_param_count,
                      raygen_params, NULL, limit);

   /* Inputs:
    * Ring offsets:                s[0-1]
    * Indirect descriptor sets:    s[2]
    * Push constants pointer:      s[3]
    * Dynamic descriptors:         s[4]
    * Traversal shader address:    s[5]
    * SBT descriptors:             s[6-7]
    * Ray launch size address:     s[8-9]
    * Dynamic callable stack base: s[10]
    * Workgroup IDs (xyz):         s[11], s[12], s[13]
    * Scratch offset:              s[14]
    * Local invocation IDs:        v[0-2]
    */
   PhysReg in_ring_offsets = get_arg_reg(in_args, in_args->ring_offsets);
   PhysReg in_descriptors = get_arg_reg(in_args, *descriptors);
   PhysReg in_push_constants = get_arg_reg(in_args, in_args->push_constants);
   PhysReg in_dynamic_descriptors = get_arg_reg(in_args, in_args->dynamic_descriptors);
   PhysReg in_sbt_desc = get_arg_reg(in_args, in_args->rt.sbt_descriptors);
   PhysReg in_traversal_addr = get_arg_reg(in_args, in_args->rt.traversal_shader_addr);
   PhysReg in_launch_size_addr = get_arg_reg(in_args, in_args->rt.launch_size_addr);
   PhysReg in_wg_id_x;
   PhysReg in_wg_id_y;
   PhysReg in_wg_id_z;
   PhysReg in_scratch_offset;
   if (options->gfx_level < GFX12) {
      in_wg_id_x = get_arg_reg(in_args, in_args->workgroup_ids[0]);
      in_wg_id_y = get_arg_reg(in_args, in_args->workgroup_ids[1]);
      in_wg_id_z = get_arg_reg(in_args, in_args->workgroup_ids[2]);
   } else {
      in_wg_id_x = PhysReg(108 + 9 /*ttmp9*/);
      in_wg_id_y = PhysReg(108 + 7 /*ttmp7*/);
   }
   if (options->gfx_level < GFX11)
      in_scratch_offset = get_arg_reg(in_args, in_args->scratch_offset);
   struct ac_arg arg_id = options->gfx_level >= GFX11 ? in_args->local_invocation_ids_packed
                                                      : in_args->local_invocation_id_x;
   PhysReg in_local_id = get_arg_reg(in_args, arg_id);

   /* Outputs:
    * Callee shader PC:            s[0-1]
    * Indirect descriptor sets:    s[2]
    * Push constants pointer:      s[3]
    * Dynamic descriptors:         s[4]
    * Traversal shader address:    s[5]
    * SBT descriptors:             s[6-7]
    * Ray launch sizes (xyz):      s[8], s[9], s[10]
    * Scratch offset (<GFX9 only): s[11]
    * Ring offsets (<GFX9 only):   s[12-13]
    * Ray launch IDs:              v[0-2]
    * Stack pointer:               v[3]
    * Shader VA:                   v[4-5]
    * Shader Record Ptr:           v[6-7]
    */
   assert(raygen_info.stack_ptr.is_reg);
   assert(raygen_info.return_address.is_reg);
   assert(raygen_info.param_infos[0].is_reg);
   assert(raygen_info.param_infos[1].is_reg);
   assert(raygen_info.param_infos[RT_ARG_LAUNCH_ID + 2].is_reg);
   assert(raygen_info.param_infos[RT_ARG_LAUNCH_SIZE + 2].is_reg);
   assert(raygen_info.param_infos[RT_ARG_DESCRIPTORS + 2].is_reg);
   assert(raygen_info.param_infos[RT_ARG_PUSH_CONSTANTS + 2].is_reg);
   assert(raygen_info.param_infos[RT_ARG_SBT_DESCRIPTORS + 2].is_reg);
   assert(raygen_info.param_infos[RAYGEN_ARG_TRAVERSAL_ADDR + 2].is_reg);
   assert(raygen_info.param_infos[RAYGEN_ARG_SHADER_RECORD_PTR + 2].is_reg);
   PhysReg out_stack_ptr_param = raygen_info.stack_ptr.def.physReg();
   PhysReg out_return_shader_addr = raygen_info.return_address.def.physReg();
   PhysReg out_divergent_shader_addr = raygen_info.param_infos[0].def.physReg();
   PhysReg out_uniform_shader_addr = raygen_info.param_infos[1].def.physReg();
   PhysReg out_launch_size_x = raygen_info.param_infos[RT_ARG_LAUNCH_SIZE + 2].def.physReg();
   PhysReg out_launch_size_y = out_launch_size_x.advance(4);
   PhysReg out_launch_size_z = out_launch_size_y.advance(4);
   PhysReg out_launch_ids[3];
   out_launch_ids[0] = raygen_info.param_infos[RT_ARG_LAUNCH_ID + 2].def.physReg();
   for (unsigned i = 1; i < 3; i++)
      out_launch_ids[i] = out_launch_ids[i - 1].advance(4);
   PhysReg out_descriptors = raygen_info.param_infos[RT_ARG_DESCRIPTORS + 2].def.physReg();
   PhysReg out_push_constants = raygen_info.param_infos[RT_ARG_PUSH_CONSTANTS + 2].def.physReg();
   PhysReg out_dynamic_descriptors =
      raygen_info.param_infos[RT_ARG_DYNAMIC_DESCRIPTORS + 2].def.physReg();
   PhysReg out_sbt_descriptors = raygen_info.param_infos[RT_ARG_SBT_DESCRIPTORS + 2].def.physReg();
   PhysReg out_traversal_addr =
      raygen_info.param_infos[RAYGEN_ARG_TRAVERSAL_ADDR + 2].def.physReg();
   PhysReg out_record_ptr = raygen_info.param_infos[RAYGEN_ARG_SHADER_RECORD_PTR + 2].def.physReg();

   unsigned param_idx = 0;
   for (auto& param_info : raygen_info.param_infos) {
      unsigned byte_size =
         align(raygen_params[param_idx].bit_size, 32) / 8 * raygen_params[param_idx].num_components;
      if (raygen_params[param_idx].is_uniform)
         num_sgprs = std::max(num_sgprs, param_info.def.physReg().reg() + byte_size / 4);
      else
         num_vgprs = std::max(num_vgprs, param_info.def.physReg().reg() - 256 + byte_size / 4);
      ++param_idx;
   }
   num_sgprs = std::max(num_sgprs, raygen_info.stack_ptr.def.physReg().reg());

   /* Temporaries: */
   PhysReg tmp_wg_start_x = PhysReg{num_sgprs};
   num_sgprs++;
   PhysReg tmp_wg_start_y = PhysReg{num_sgprs};
   num_sgprs++;
   PhysReg tmp_swizzle_bound_y = PhysReg{num_sgprs};
   num_sgprs++;
   PhysReg tmp_wg_id_y = PhysReg{num_sgprs};
   num_sgprs++;
   num_sgprs = align(num_sgprs, 2);
   PhysReg tmp_raygen_sbt = PhysReg{num_sgprs};
   num_sgprs += 2;
   PhysReg tmp_launch_size_addr = PhysReg{num_sgprs};
   num_sgprs += 2;
   PhysReg tmp_ring_offsets = PhysReg{num_sgprs};
   num_sgprs += 2;
   PhysReg tmp_sbt_desc = PhysReg{num_sgprs};
   if (program->gfx_level < GFX9)
      num_sgprs += 2;
   PhysReg tmp_traversal_addr = PhysReg{num_sgprs};
   num_sgprs += 1;
   PhysReg tmp_push_constants = PhysReg{num_sgprs};
   num_sgprs++;
   PhysReg tmp_descriptors = PhysReg{num_sgprs};
   num_sgprs++;
   PhysReg tmp_dynamic_descriptors = PhysReg{num_sgprs};
   num_sgprs++;

   PhysReg tmp_swizzled_id_x = PhysReg{256 + num_vgprs++};
   PhysReg tmp_swizzled_id_y = PhysReg{256 + num_vgprs++};
   PhysReg tmp_swizzled_id_shifted_x = PhysReg{256 + num_vgprs++};
   PhysReg tmp_swizzled_id_shifted_y = PhysReg{256 + num_vgprs++};

   /* Confirm some assumptions about register aliasing */
   if (program->gfx_level >= GFX9) {
      if (program->gfx_level < GFX12) {
         assert(in_wg_id_z == out_launch_size_y);
         assert(in_wg_id_y == out_launch_size_x);
      }
      assert(in_sbt_desc == out_sbt_descriptors);
      assert(in_traversal_addr == out_descriptors);
   } else {
      assert(out_launch_size_x == in_wg_id_y);
      assert(out_sbt_descriptors == in_launch_size_addr);
   }

   /* load raygen sbt */
   bld.smem(aco_opcode::s_load_dwordx2, Definition(tmp_raygen_sbt, s2), Operand(in_sbt_desc, s2),
            Operand::c32(0u));

   bld.sop1(aco_opcode::s_mov_b64, Definition(tmp_launch_size_addr, s2),
            Operand(in_launch_size_addr, s2));
   bld.sop1(aco_opcode::s_mov_b32, Definition(tmp_traversal_addr, s1),
            Operand(in_traversal_addr, s1));

   /* On GFX8-, the out push constant/descriptor parameters alias WG IDs, so we copy these
    * parameters only after we're done calculating the launch IDs.
    */
   bld.sop1(aco_opcode::s_mov_b32, Definition(tmp_push_constants, s1),
            Operand(in_push_constants, s1));
   bld.sop1(aco_opcode::s_mov_b32, Definition(tmp_dynamic_descriptors, s1),
            Operand(in_dynamic_descriptors, s1));
   bld.sop1(aco_opcode::s_mov_b32, Definition(tmp_descriptors, s1), Operand(in_descriptors, s1));

   if (options->gfx_level < GFX9)
      bld.sop1(aco_opcode::s_mov_b64, Definition(tmp_sbt_desc, s2), Operand(in_sbt_desc, s2));

   /* init scratch */
   if (options->gfx_level < GFX9) {
      /* Unconditionally apply the scratch offset to scratch_rsrc so we just have
       * to pass the rsrc through to callees.
       */
      bld.sop2(aco_opcode::s_add_u32, Definition(tmp_ring_offsets, s1), Definition(scc, s1),
               Operand(in_ring_offsets, s1), Operand(in_scratch_offset, s1));
      bld.sop2(aco_opcode::s_addc_u32, Definition(tmp_ring_offsets.advance(4), s1),
               Definition(scc, s1), Operand(in_ring_offsets.advance(4), s1), Operand::c32(0),
               Operand(scc, s1));
   } else if (options->gfx_level < GFX11) {
      hw_init_scratch(bld, Definition(in_ring_offsets, s1), Operand(in_ring_offsets, s2),
                      Operand(in_scratch_offset, s1));
   }

   /* Set up the Z launch ID, as well as setting up workgroup Y IDs. On gfx11-, the setup consists
    * of backing the ID up as the load for the ray launch sizes will overwrite it.
    */
   if (options->gfx_level >= GFX12) {
      bld.vop2_e64(aco_opcode::v_lshrrev_b32, Definition(out_launch_ids[2], v1), Operand::c32(16),
                   Operand(in_wg_id_y, s1));
      bld.sop2(aco_opcode::s_pack_ll_b32_b16, Definition(tmp_wg_id_y, s1), Operand(in_wg_id_y, s1),
               Operand::c32(0));
   } else {
      bld.vop1(aco_opcode::v_mov_b32, Definition(out_launch_ids[2], v1), Operand(in_wg_id_z, s1));
      bld.sop1(aco_opcode::s_mov_b32, Definition(tmp_wg_id_y, s1), Operand(in_wg_id_y, s1));
   }

   /* load raygen address */
   bld.smem(aco_opcode::s_load_dwordx2, Definition(out_uniform_shader_addr, s2),
            Operand(tmp_raygen_sbt, s2), Operand::c32(0u));

   /* load ray launch sizes */
   assert(out_launch_size_x.reg() % 4 == 0);
   if (options->gfx_level >= GFX12) {
      bld.smem(aco_opcode::s_load_dwordx3, Definition(out_launch_size_x, s3),
               Operand(tmp_launch_size_addr, s2), Operand::c32(0u));
   } else {
      bld.smem(aco_opcode::s_load_dword, Definition(out_launch_size_z, s1),
               Operand(tmp_launch_size_addr, s2), Operand::c32(8u));
      bld.smem(aco_opcode::s_load_dwordx2, Definition(out_launch_size_x, s2),
               Operand(tmp_launch_size_addr, s2), Operand::c32(0u));
   }

   /* Swizzle ray launch IDs. We dispatch a 1D 32x1/64x1 workgroup natively. Many games dispatch
    * rays in a 2D grid and write RT results to an image indexed by the x/y launch ID.
    * In image space, a 1D workgroup maps to a 32/64-pixel wide line, which is inefficient for two
    * reasons:
    * - Image data is usually arranged on a Z-order curve, a long line makes for inefficient
    *   memory access patterns.
    * - Each wave working on a "line" in image space may increase divergence. It's better to trace
    *   rays in a small square, since that makes it more likely all rays hit the same or similar
    *   objects.
    *
    * It turns out arranging rays along a Z-order curve is best for both image access patterns and
    * ray divergence. Since image data is swizzled along a Z-order curve as well, swizzling the
    * launch ID should result in each lane accessing whole cachelines at once. For traced rays,
    * the Z-order curve means that each quad is arranged in a 2x2 square in image space as well.
    * Since the RT unit processes 4 lanes at a time, reducing divergence per quad may result in
    * better RT unit utilization (for example by the RT unit being able to skip the quad entirely
    * if all 4 lanes are inactive).
    *
    * To swizzle along a Z-order curve, treat the 1D lane ID as a morton code. Then, do the inverse
    * of morton code generation (i.e. deinterleaving the bits) to recover the x-y
    * coordinates on the Z-order curve.
    */

   /* Deinterleave bits - even bits go to tmp_swizzled_id_x, odd ones to tmp_swizzled_id_y */
   bld.vop2(aco_opcode::v_lshrrev_b32, Definition(tmp_swizzled_id_y, v1), Operand::c32(1),
            Operand(in_local_id, v1));

   /* The deinterleaved bits are currently separated by single bit, like so:
    * ...0 0 0 A ? B ? C
    * Compact the deinterleaved bits by factor 2 to remove the padding, resulting in
    * ...0 0 0 0 0 A B C
    */
   bld.vop2(aco_opcode::v_lshrrev_b32, Definition(tmp_swizzled_id_shifted_y, v1), Operand::c32(1),
            Operand(tmp_swizzled_id_y, v1));
   /* Use tmp_swizzled_id_y instead of creating a tmp_swizzled_id_shifted_x, since they would both
    * be in_local_id>>1 */
   bld.vop3(aco_opcode::v_bfi_b32, Definition(tmp_swizzled_id_x, v1), Operand::c32(0x11),
            Operand(in_local_id, v1), Operand(tmp_swizzled_id_y, v1));
   bld.vop3(aco_opcode::v_bfi_b32, Definition(tmp_swizzled_id_y, v1), Operand::c32(0x11),
            Operand(tmp_swizzled_id_y, v1), Operand(tmp_swizzled_id_shifted_y, v1));

   bld.vop2(aco_opcode::v_lshrrev_b32, Definition(tmp_swizzled_id_shifted_x, v1), Operand::c32(2),
            Operand(tmp_swizzled_id_x, v1));
   bld.vop2(aco_opcode::v_lshrrev_b32, Definition(tmp_swizzled_id_shifted_y, v1), Operand::c32(2),
            Operand(tmp_swizzled_id_y, v1));
   bld.vop3(aco_opcode::v_bfi_b32, Definition(tmp_swizzled_id_x, v1), Operand::c32(0x3),
            Operand(tmp_swizzled_id_x, v1), Operand(tmp_swizzled_id_shifted_x, v1));
   bld.vop3(aco_opcode::v_bfi_b32, Definition(tmp_swizzled_id_y, v1), Operand::c32(0x3),
            Operand(tmp_swizzled_id_y, v1), Operand(tmp_swizzled_id_shifted_y, v1));

   /* Fix up the workgroup IDs after converting from 32x1/64x1 to 8x4/8x8. The X dimension of the
    * workgroup size gets divided by 4/8, while the Y dimension gets multiplied by the same amount.
    * Rearrange the workgroups to make up for that, by rounding the Y component of the workgroup ID
    * to the nearest multiple of 4/8. The remainder gets added to the X dimension, to make up for
    * the fact we divided the X component of the ID.
    */
   uint32_t workgroup_size_log2 = util_logbase2(program->workgroup_size);
   bld.sop2(aco_opcode::s_lshl_b32, Definition(tmp_wg_start_x, s1), Definition(scc, s1),
            Operand(in_wg_id_x, s1), Operand::c32(workgroup_size_log2));

   /* unsigned y_remainder = tmp_wg_id_y % wg_height
    * We use tmp_wg_start_y to store y_rem, and overwrite it later with the real wg_start_y.
    */
   uint32_t workgroup_width_log2 = 3u;
   uint32_t workgroup_height_mask = program->workgroup_size == 32 ? 0x3u : 0x7u;
   bld.sop2(aco_opcode::s_and_b32, Definition(tmp_wg_start_y, s1), Definition(scc, s1),
            Operand(tmp_wg_id_y, s1), Operand::c32(workgroup_height_mask));
   /* wg_start_x += y_remainder * workgroup_width (workgroup_width == 8) */
   bld.sop2(aco_opcode::s_lshl_b32, Definition(tmp_wg_start_y, s1), Definition(scc, s1),
            Operand(tmp_wg_start_y, s1), Operand::c32(workgroup_width_log2));
   bld.sop2(aco_opcode::s_add_u32, Definition(tmp_wg_start_x, s1), Definition(scc, s1),
            Operand(tmp_wg_start_x, s1), Operand(tmp_wg_start_y, s1));
   /* wg_start_y = ROUND_DOWN_TO(in_wg_y, workgroup_height) */
   bld.sop2(aco_opcode::s_and_b32, Definition(tmp_wg_start_y, s1), Definition(scc, s1),
            Operand(tmp_wg_id_y, s1), Operand::c32(~workgroup_height_mask));

   if (options->gfx_level < GFX9) {
      bld.vop2(aco_opcode::v_add_co_u32, Definition(tmp_swizzled_id_x, v1), Definition(vcc, s2),
               Operand(tmp_wg_start_x, s1), Operand(tmp_swizzled_id_x, v1));
      bld.vop2(aco_opcode::v_add_co_u32, Definition(tmp_swizzled_id_y, v1), Definition(vcc, s2),
               Operand(tmp_wg_start_y, s1), Operand(tmp_swizzled_id_y, v1));
   } else {
      bld.vop2(aco_opcode::v_add_u32, Definition(tmp_swizzled_id_x, v1), Operand(tmp_wg_start_x, s1),
               Operand(tmp_swizzled_id_x, v1));
      bld.vop2(aco_opcode::v_add_u32, Definition(tmp_swizzled_id_y, v1), Operand(tmp_wg_start_y, s1),
               Operand(tmp_swizzled_id_y, v1));
   }

   /* We can only swizzle launch IDs if we run a full workgroup, and the resulting launch IDs
    * won't exceed the launch size. Calculate unswizzled launch IDs here to fall back to them
    * if the swizzled launch IDs are out of bounds.
    */
   bld.vop1(aco_opcode::v_mov_b32, Definition(out_launch_ids[1], v1), Operand(tmp_wg_id_y, s1));
   bld.vop3(aco_opcode::v_mad_u32_u24, Definition(out_launch_ids[0], v1), Operand(in_wg_id_x, s1),
            Operand::c32(program->workgroup_size), Operand(in_local_id, v1));

   /* Round the launch size down to the nearest multiple of workgroup_height. If the workgroup ID
    * exceeds this, then the swizzled IDs' Y component will exceed the Y launch size and we have to
    * fall back to unswizzled IDs.
    */
   bld.sop2(aco_opcode::s_and_b32, Definition(tmp_swizzle_bound_y, s1), Definition(scc, s1),
            Operand(out_launch_size_y, s1), Operand::c32(~workgroup_height_mask));
   /* If we are only running a partial workgroup, swizzling would yield a wrong result. */
   if (program->gfx_level >= GFX8) {
      bld.sopc(Builder::s_cmp_lg, Definition(scc, s1), Operand(exec, bld.lm),
               Operand::c32_or_c64(-1u, program->workgroup_size == 64));
   } else {
      /* Write the XOR result to vcc because it's currently unused and a convenient register (always
       * the same size as exec). We only care about the value of scc, i.e. if the result is nonzero
       * (vcc is about to be overwritten anyway).
       */
      bld.sop2(Builder::s_xor, Definition(vcc, bld.lm), Definition(scc, s1), Operand(exec, bld.lm),
               Operand::c32_or_c64(-1u, program->workgroup_size == 64));
   }
   bld.sop2(Builder::s_cselect, Definition(vcc, bld.lm),
            Operand::c32_or_c64(-1u, program->wave_size == 64),
            Operand::c32_or_c64(0u, program->wave_size == 64), Operand(scc, s1));
   bld.sopc(aco_opcode::s_cmp_ge_u32, Definition(scc, s1), Operand(tmp_wg_id_y, s1),
            Operand(tmp_swizzle_bound_y, s1));
   bld.sop2(Builder::s_cselect, Definition(vcc, bld.lm),
            Operand::c32_or_c64(-1u, program->wave_size == 64),
            Operand(vcc, bld.lm), Operand(scc, s1));

   bld.vop2(aco_opcode::v_cndmask_b32, Definition(out_launch_ids[0], v1),
            Operand(tmp_swizzled_id_x, v1), Operand(out_launch_ids[0], v1), Operand(vcc, bld.lm));
   bld.vop2(aco_opcode::v_cndmask_b32, Definition(out_launch_ids[1], v1),
            Operand(tmp_swizzled_id_y, v1), Operand(out_launch_ids[1], v1), Operand(vcc, bld.lm));

   /* calculate shader record ptr: SBT + RADV_RT_HANDLE_SIZE */
   if (options->gfx_level < GFX9) {
      bld.vop2_e64(aco_opcode::v_add_co_u32, Definition(out_record_ptr, v1), Definition(vcc, s2),
                   Operand(tmp_raygen_sbt, s1), Operand::c32(32u));
   } else {
      bld.vop2_e64(aco_opcode::v_add_u32, Definition(out_record_ptr, v1),
                   Operand(tmp_raygen_sbt, s1), Operand::c32(32u));
   }
   bld.vop1(aco_opcode::v_mov_b32, Definition(out_record_ptr.advance(4), v1),
            Operand(tmp_raygen_sbt.advance(4), s1));

   bld.sop1(aco_opcode::s_mov_b32, Definition(out_traversal_addr, s1),
            Operand(tmp_traversal_addr, s1));
   bld.sop1(aco_opcode::s_mov_b32, Definition(out_traversal_addr.advance(4), s1),
            Operand::c32(options->address32_hi));

   if (program->gfx_level < GFX8)
      bld.vop3(aco_opcode::v_lshr_b64, Definition(out_divergent_shader_addr, v2),
               Operand(out_uniform_shader_addr, s2), Operand::c32(0));
   else
      bld.vop3(aco_opcode::v_lshrrev_b64, Definition(out_divergent_shader_addr, v2),
               Operand::c32(0), Operand(out_uniform_shader_addr, s2));

   /* Launch IDs are calculated, so copy the push constant/sbt descriptor parameters.
    * Do this here before other parameters overwrite the inputs.
    */
   if (program->gfx_level < GFX9) {
      bld.sop1(aco_opcode::s_mov_b32, Definition(out_sbt_descriptors, s1),
               Operand(tmp_sbt_desc, s1));
      bld.sop1(aco_opcode::s_mov_b32, Definition(out_sbt_descriptors.advance(4), s1),
               Operand(tmp_sbt_desc.advance(4), s1));
   }
   bld.sop1(aco_opcode::s_mov_b32, Definition(out_push_constants, s1),
            Operand(tmp_push_constants, s1));
   bld.sop1(aco_opcode::s_mov_b32, Definition(out_dynamic_descriptors, s1),
            Operand(tmp_dynamic_descriptors, s1));
   bld.sop1(aco_opcode::s_mov_b32, Definition(out_descriptors, s1), Operand(tmp_descriptors, s1));

   bld.sop1(aco_opcode::s_mov_b64, Definition(out_return_shader_addr, s2), Operand::c32(0));

   if (program->gfx_level >= GFX9) {
      bld.sopk(aco_opcode::s_movk_i32, Definition(out_stack_ptr_param, s1), 0);
   } else {
      /* Construct the scratch_rsrc here and pass it to the callees to use directly. */
      struct ac_buffer_state ac_state = {0};
      uint32_t desc[4];

      ac_state.size = 0xffffffff;
      ac_state.format = PIPE_FORMAT_R32_FLOAT;
      for (int i = 0; i < 4; i++)
         ac_state.swizzle[i] = PIPE_SWIZZLE_0;
      ac_state.element_size = 1u;
      ac_state.index_stride = program->wave_size == 64 ? 3u : 2u;
      ac_state.add_tid = true;
      ac_state.gfx10_oob_select = V_008F0C_OOB_SELECT_RAW;

      ac_build_buffer_descriptor(program->gfx_level, &ac_state, desc);

      bld.sop1(aco_opcode::s_mov_b32, Definition(out_stack_ptr_param, s1),
               Operand(tmp_ring_offsets, s1));
      bld.sop1(aco_opcode::s_mov_b32, Definition(out_stack_ptr_param.advance(4), s1),
               Operand(tmp_ring_offsets.advance(4), s1));
      bld.sop1(aco_opcode::s_mov_b32, Definition(out_stack_ptr_param.advance(8), s1),
               Operand::c32(desc[2]));
      bld.sop1(aco_opcode::s_mov_b32, Definition(out_stack_ptr_param.advance(12), s1),
               Operand::c32(desc[3]));
   }

   /* jump to raygen */
   bld.sop1(aco_opcode::s_setpc_b64, Operand(out_uniform_shader_addr, s2));

   program->config->float_mode = program->blocks[0].fp_mode.val;
   program->config->num_vgprs = get_vgpr_alloc(program, num_vgprs);
   program->config->num_sgprs = get_sgpr_alloc(program, num_sgprs);
}

} // namespace aco
