/*
 * Copyright © 2025 Red Hat
 *
 * SPDX-License-Identifier: MIT
 */
#include "lvp_nir.h"

extern unsigned lp_native_vector_width;

#define MAX_CMAT_LEN 16
#define CMAT_LEN (lp_native_vector_width / 32)

/* This pass lowers cooperative matrix.
 *
 * for lavapipe we advertise 8x8 matrix.
 * This means we can store vec8[8] and get the backend to do the right thing.
 */
static unsigned
get_cmat_size(struct glsl_cmat_description matrix_desc)
{
   return matrix_desc.cols * matrix_desc.rows;
}

static unsigned
get_cmat_length(struct glsl_cmat_description matrix_desc)
{
   return get_cmat_size(matrix_desc) / CMAT_LEN;
}

static const struct glsl_type *
remap_matrix_type(struct hash_table *mapping, const struct glsl_type *orig)
{
  struct hash_entry *entry = _mesa_hash_table_search(mapping, orig);

   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;

   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description matrix_desc =
         *glsl_get_cmat_description(orig);

      new_type = glsl_vector_type(matrix_desc.element_type, get_cmat_length(matrix_desc));
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *elem_type = glsl_get_array_element(orig);
      const struct glsl_type *new_elem_type =
         remap_matrix_type(mapping, elem_type);

      if (elem_type != new_elem_type) {
         new_type = glsl_array_type(new_elem_type, glsl_get_length(orig),
                                    glsl_get_explicit_stride(orig));
      }
   }
   _mesa_hash_table_insert(mapping, orig, (void *)new_type);
   return new_type;
}

static nir_def *
load_cmat_deref(nir_builder *b, nir_deref_instr *src)
{
   struct glsl_cmat_description matrix_desc =
      *glsl_get_cmat_description(src->type);

   return nir_build_load_deref(
      b, get_cmat_length(matrix_desc),
      glsl_base_type_bit_size(matrix_desc.element_type), &src->def, 0);
}

static ALWAYS_INLINE nir_def *
load_cmat_src(nir_builder *b, nir_src src)
{
   return load_cmat_deref(b, nir_src_as_deref(src));
}

static ALWAYS_INLINE struct glsl_cmat_description
cmat_src_desc(nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   return *glsl_get_cmat_description(deref->type);
}

static void
store_cmat_deref(nir_builder *b, nir_deref_instr *dst, nir_def *val)
{
   ASSERTED struct glsl_cmat_description matrix_desc =
      *glsl_get_cmat_description(dst->type);

   assert(val->bit_size == glsl_base_type_bit_size(matrix_desc.element_type));
   assert(val->num_components == get_cmat_length(matrix_desc));

   nir_store_deref(b, dst, val, ~0);
}

static ALWAYS_INLINE void
store_cmat_src(nir_builder *b, nir_src dst_src, nir_def *val)
{
   store_cmat_deref(b, nir_src_as_deref(dst_src), val);
}

static bool
lower_cmat_copy(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_build_copy_deref(b, intr->src[0].ssa, intr->src[1].ssa);
   nir_instr_remove(&intr->instr);
   return true;
}

static nir_def *
convert_base_type(nir_builder *b, nir_def *src, enum glsl_base_type src_type, enum glsl_base_type dst_type)
{
   if (dst_type == src_type)
      return src;

   nir_op op = nir_type_conversion_op(nir_get_nir_type_for_glsl_base_type(src_type),
                                      nir_get_nir_type_for_glsl_base_type(dst_type), nir_rounding_mode_undef);

   return nir_build_alu1(b, op, src);
}

static bool
lower_cmat_convert(nir_builder *b,
                   nir_intrinsic_instr *intr)
{
   struct glsl_cmat_description dst_desc = cmat_src_desc(intr->src[0]);
   struct glsl_cmat_description src_desc = cmat_src_desc(intr->src[1]);

   enum glsl_base_type dst_element_type = dst_desc.element_type;
   enum glsl_base_type src_element_type = src_desc.element_type;
   nir_def *cmat = load_cmat_src(b, intr->src[1]);

   nir_def *ret = convert_base_type(b, cmat, src_element_type, dst_element_type);
   store_cmat_src(b, intr->src[0], ret);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_load_store(nir_builder *b,
                      struct hash_table *type_mapping,
                      nir_intrinsic_instr *intr)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[!is_load]);
   enum glsl_matrix_layout layout = nir_intrinsic_matrix_layout(intr);
   nir_deref_instr *cmat_deref = nir_src_as_deref(intr->src[!is_load]);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[is_load]);
   nir_def *stride = intr->src[2].ssa;

   nir_def *lane_id = nir_load_subgroup_invocation(b);
   unsigned type_size_B = glsl_base_type_bit_size(desc.element_type) / 8;
   const uint32_t ptr_stride = glsl_get_bit_size(deref->type) / 8 * glsl_get_vector_elements(deref->type);
   deref = nir_build_deref_cast(b, &deref->def, deref->modes, deref->type, ptr_stride);
   const struct glsl_type *cmat_type = remap_matrix_type(type_mapping, cmat_deref->type);
   cmat_deref = nir_build_deref_cast(b, &cmat_deref->def, cmat_deref->modes,
                                     cmat_type, 0);

   /* store B matrix transposed */
   if (desc.use == GLSL_CMAT_USE_B)
      layout =
         layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR ? GLSL_MATRIX_LAYOUT_ROW_MAJOR : GLSL_MATRIX_LAYOUT_COLUMN_MAJOR;

   unsigned idx_bits = deref->def.bit_size;
   nir_def *vars[MAX_CMAT_LEN];

   if (!is_load) {
      nir_def *src = load_cmat_src(b, intr->src[!is_load]);
      for (unsigned i = 0; i < CMAT_LEN; i++) {
         vars[i] = nir_channel(b, src, i);
      }
   }

   for (unsigned i = 0; i < CMAT_LEN; i++) {
      nir_def *col_offset = lane_id;
      nir_def *row_offset = nir_imm_int(b, i);

      if (layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
         SWAP(col_offset, row_offset);
      }

      col_offset = nir_imul(b, col_offset, stride);
      col_offset = nir_u2uN(b, col_offset, idx_bits);
      row_offset = nir_u2uN(b, row_offset, idx_bits);

      nir_deref_instr *iter_deref = nir_build_deref_ptr_as_array(b, deref, col_offset);

      iter_deref = nir_build_deref_cast(b, &iter_deref->def,
                                        deref->modes,
                                        glsl_scalar_type(desc.element_type),
                                        type_size_B);
      iter_deref = nir_build_deref_ptr_as_array(b, iter_deref, row_offset);

      if (is_load) {
         vars[i] = nir_load_deref(b, iter_deref);
      } else {
         nir_store_deref(b, iter_deref, vars[i], ~0);
      }
   }

   if (is_load) {
      nir_def *mat = nir_vec(b, vars, CMAT_LEN);
      nir_store_deref(b, cmat_deref, mat, nir_component_mask(mat->num_components));
   }
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_construct(nir_builder *b,
                     nir_intrinsic_instr *intr)
{
   nir_deref_instr *dst_deref = nir_src_as_deref(intr->src[0]);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(dst_deref->type);
   nir_def *elem = intr->src[1].ssa;

   nir_def *r = nir_replicate(b, elem, get_cmat_length(desc));

   nir_store_deref(b, dst_deref, r, nir_component_mask(r->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_extract(nir_builder *b,
                     nir_intrinsic_instr *intr)
{
   nir_def *mat = load_cmat_src(b, intr->src[0]);
   nir_def *index = intr->src[1].ssa;
   nir_def *elem = nir_vector_extract(b, mat, index);
   nir_def_replace(&intr->def, elem);
   return true;
}

static bool
lower_cmat_insert(nir_builder *b,
                  nir_intrinsic_instr *intr)
{
   nir_def *elem = intr->src[1].ssa;
   nir_def *mat = load_cmat_src(b, intr->src[2]);
   nir_def *index = intr->src[3].ssa;

   nir_def *r = nir_vector_insert(b, mat, elem, index);
   store_cmat_src(b, intr->src[0], r);

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_binary_op(nir_builder *b,
                     nir_intrinsic_instr *intr)
{
   nir_def *src_a = load_cmat_src(b, intr->src[1]);
   nir_def *src_b = load_cmat_src(b, intr->src[2]);
   nir_op op = nir_intrinsic_alu_op(intr);

   nir_def *ret = nir_build_alu2(b, op, src_a, src_b);
   store_cmat_src(b, intr->src[0], ret);

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_unary_op(nir_builder *b,
                     nir_intrinsic_instr *intr)
{
   nir_def *src = load_cmat_src(b, intr->src[1]);
   nir_op op = nir_intrinsic_alu_op(intr);

   nir_def *ret = nir_build_alu1(b, op, src);
   store_cmat_src(b, intr->src[0], ret);

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_scalar_op(nir_builder *b,
                     nir_intrinsic_instr *intr)
{
   nir_def *src_a = load_cmat_src(b, intr->src[1]);
   nir_op op = nir_intrinsic_alu_op(intr);

   nir_def *ret = nir_build_alu2(b, op, src_a, intr->src[2].ssa);
   store_cmat_src(b, intr->src[0], ret);

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_length(nir_builder *b,
                  nir_intrinsic_instr *intr)
{
   nir_def_replace(&intr->def, nir_imm_int(b, CMAT_LEN));
   return true;
}

static bool
lower_cmat_muladd(nir_builder *b,
                  nir_intrinsic_instr *intr)
{
   const struct glsl_cmat_description a_desc = cmat_src_desc(intr->src[1]);
   const struct glsl_cmat_description b_desc = cmat_src_desc(intr->src[2]);
   const struct glsl_cmat_description c_desc = cmat_src_desc(intr->src[3]);
   nir_def *cmat_a = load_cmat_src(b, intr->src[1]);
   nir_def *cmat_b = load_cmat_src(b, intr->src[2]);
   nir_def *cmat_c = load_cmat_src(b, intr->src[3]);

   unsigned a_length = get_cmat_length(a_desc);
   unsigned b_length = get_cmat_length(b_desc);
   unsigned c_length = get_cmat_length(c_desc);
   nir_def *a_comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *b_comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *c_comps[NIR_MAX_VEC_COMPONENTS];
   nir_def *d_comps[NIR_MAX_VEC_COMPONENTS];
   const nir_cmat_signed cmat_signed_mask = nir_intrinsic_cmat_signed_mask(intr);

   enum glsl_base_type c_element_type =
      glsl_apply_signedness_to_base_type(c_desc.element_type, cmat_signed_mask & NIR_CMAT_C_SIGNED);

   for (unsigned i = 0; i < a_length; i++)
      a_comps[i] = nir_channel(b, cmat_a, i);

   for (unsigned i = 0; i < b_length; i++)
      b_comps[i] = nir_channel(b, cmat_b, i);

   for (unsigned i = 0; i < c_length; i++)
      c_comps[i] = nir_channel(b, cmat_c, i);

   nir_def *lane_id = nir_load_subgroup_invocation(b);
   int accum_bit_size = glsl_base_type_bit_size(c_desc.element_type);
   for (unsigned i = 0; i < CMAT_LEN; i++) {
      nir_def *ref = nir_imm_zero(b, 1, glsl_base_type_bit_size(c_desc.element_type));
      for (unsigned j = 0; j < CMAT_LEN; j++) {
         nir_def *outer_else_val = ref;
         ref = nir_imm_zero(b, 1, glsl_base_type_bit_size(c_desc.element_type));

         nir_def *a_i = a_comps[i];
         nir_def *b_j = b_comps[j]; /* B is stored transposed */
         nir_def *val;
         if (glsl_base_type_is_integer(c_desc.element_type)) {
            if (c_element_type == GLSL_TYPE_INT)
               a_i = nir_i2iN(b, a_i, accum_bit_size);
            else
               a_i = nir_u2uN(b, a_i, accum_bit_size);
            if (c_element_type == GLSL_TYPE_INT)
               b_j = nir_i2iN(b, b_j, accum_bit_size);
            else
               b_j = nir_u2uN(b, b_j, accum_bit_size);

            val = nir_imul(b, a_i, b_j);
            ref = nir_iadd(b, ref, val);
         } else {
            a_i = nir_f2fN(b, a_i, accum_bit_size);
            b_j = nir_f2fN(b, b_j, accum_bit_size);
            val = nir_fmul(b, a_i, b_j);
            ref = nir_fadd(b, ref, val);
         }

         if (glsl_base_type_is_integer(c_desc.element_type)) {
            ref = nir_reduce(b, ref, .reduction_op = nir_op_iadd);
         } else {
            ref = nir_reduce(b, ref, .reduction_op = nir_op_fadd);
         }

         nir_def *lane = nir_ieq_imm(b, lane_id, j);
         ref = nir_bcsel(b, lane, ref, outer_else_val);
      }

      if (glsl_base_type_is_integer(c_desc.element_type)) {
         ref = nir_iadd(b, ref, c_comps[i]);
      } else {
         ref = nir_fadd(b, ref, c_comps[i]);
      }
      d_comps[i] = ref;
   }
   nir_def *ret = nir_vec(b, d_comps, CMAT_LEN);
   store_cmat_src(b, intr->src[0], ret);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_bitcast(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *src1 = load_cmat_src(b, intr->src[1]);
   nir_store_deref(b, nir_src_as_deref(intr->src[0]), src1, nir_component_mask(src1->num_components));
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_impl(nir_function_impl *impl,
           struct hash_table *type_mapping)
{
   bool progress = false;
   /* Remap all cmat temp var to array of scalars */
   nir_foreach_function_temp_variable(var, impl) {
      const struct glsl_type *new_type =
         remap_matrix_type(type_mapping, var->type);
      if (new_type != var->type) {
         var->type = new_type;
         progress = true;
      }
   }

   /* Iterate in reverse order so that lowering can still use the matrix types from the derefs before we change it. */
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe (block, impl) {
      nir_foreach_instr_reverse_safe (instr, block) {
         b.cursor = nir_before_instr(instr);

         switch (instr->type) {
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_cmat_length:
               progress |= lower_cmat_length(&b, intr);
               break;
            case nir_intrinsic_cmat_construct:
               progress |= lower_cmat_construct(&b, intr);
               break;
            case nir_intrinsic_cmat_extract:
               progress |= lower_cmat_extract(&b, intr);
               break;
            case nir_intrinsic_cmat_insert:
               progress |= lower_cmat_insert(&b, intr);
               break;
            case nir_intrinsic_cmat_load:
            case nir_intrinsic_cmat_store:
               progress |= lower_cmat_load_store(&b, type_mapping, intr);
               break;
            case nir_intrinsic_cmat_binary_op:
               progress |= lower_cmat_binary_op(&b, intr);
               break;
            case nir_intrinsic_cmat_unary_op:
               progress |= lower_cmat_unary_op(&b, intr);
               break;
            case nir_intrinsic_cmat_scalar_op:
               progress |= lower_cmat_scalar_op(&b, intr);
               break;
            case nir_intrinsic_cmat_muladd:
               progress |= lower_cmat_muladd(&b, intr);
               break;
            case nir_intrinsic_cmat_copy:
               progress |= lower_cmat_copy(&b, intr);
               break;
            case nir_intrinsic_cmat_convert:
               progress |= lower_cmat_convert(&b, intr);
               break;
            case nir_intrinsic_cmat_bitcast:
               progress |= lower_cmat_bitcast(&b, intr);
               break;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            const struct glsl_type *new_type =
               remap_matrix_type(type_mapping, deref->type);

            if (new_type != deref->type) {
               deref->type = new_type;
               progress = true;
            }
            break;
         }
         default:
            break;
         }
      }
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

bool
lvp_nir_lower_cooperative_matrix(nir_shader *shader)
{
   bool progress = false;

   if (!shader->info.cs.has_cooperative_matrix)
      return false;

   struct hash_table *type_mapping = _mesa_pointer_hash_table_create(NULL);
   /* Remap all cmat shader temp var to array of vectors */
   nir_foreach_variable_with_modes(var, shader, nir_var_shader_temp) {
      const struct glsl_type *new_type =
         remap_matrix_type(type_mapping, var->type);

      if (new_type != var->type) {
         var->type = new_type;
         progress = true;
      }
   }

   progress |= lower_impl(nir_shader_get_entrypoint(shader), type_mapping);

   _mesa_hash_table_destroy(type_mapping, NULL);
   return progress;
}
