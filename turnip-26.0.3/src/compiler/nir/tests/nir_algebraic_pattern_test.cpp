/*
 * Copyright © 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "nir_algebraic_pattern_test.h"
#include "nir_builder.h"
#include "nir_constant_expressions.h"

#include "util/memstream.h"

#include <float.h>
#include <math.h>

nir_algebraic_pattern_test::nir_algebraic_pattern_test(const char *name)
    : nir_test(name)
{
}

static nir_const_value *
tmp_value(nir_algebraic_pattern_test *test, nir_def *def)
{
   return &test->tmp_values[def->index * NIR_MAX_VEC_COMPONENTS];
}

static bool
def_annotate_value(nir_def *def, void *data)
{
   nir_algebraic_pattern_test *test = (nir_algebraic_pattern_test *)data;

   char *annotation = NULL;
   size_t annotation_size = 0;
   u_memstream mem;
   if (!u_memstream_open(&mem, &annotation, &annotation_size))
      return true;

   FILE *output = u_memstream_get(&mem);

   nir_const_value *value = tmp_value(test, def);

   fprintf(output, "// ");
   if (def->num_components == 1) {
      fprintf(output, "0x%0" PRIx64, value->u64);
      if (def->bit_size >= 16)
         fprintf(output, " = %f", nir_const_value_as_float(value[0], def->bit_size));

   } else {
      fprintf(output, "(");
      for (uint32_t comp = 0; comp < def->num_components; comp++) {
         if (comp > 0)
            fprintf(output, ", ");
         fprintf(output, "0x%0" PRIx64, value[comp].u64);
      }
      fprintf(output, ")");
      if (def->bit_size >= 16) {
         fprintf(output, " = (");
         for (uint32_t comp = 0; comp < def->num_components; comp++) {
            if (comp > 0)
               fprintf(output, ", ");
            fprintf(output, "%f", nir_const_value_as_float(value[comp], def->bit_size));
         }
         fprintf(output, ")");
      }
   }

   fputc(0, output);

   u_memstream_close(&mem);

   _mesa_hash_table_insert(test->annotations, nir_def_instr(def), annotation);

   return true;
}

static bool
instr_annotate_value(nir_builder *b, nir_instr *instr, void *data)
{
   nir_foreach_def(instr, def_annotate_value, data);
   return false;
}

nir_algebraic_pattern_test::~nir_algebraic_pattern_test()
{
   if (HasFailure()) {
      annotations = _mesa_pointer_hash_table_create(nullptr);
      nir_shader_instructions_pass(b->shader, instr_annotate_value, nir_metadata_all, this);
   }
}

static bool
count_input(nir_builder *b, nir_intrinsic_instr *intrinsic, void *data)
{
   nir_algebraic_pattern_test *test = (nir_algebraic_pattern_test *)data;

   if (intrinsic->intrinsic == nir_intrinsic_unit_test_uniform_input)
      test->input_count = MAX2(test->input_count, (uint32_t)nir_intrinsic_base(intrinsic) + 1);

   return false;
}

static bool
nir_def_is_used_as(nir_def *def, nir_alu_type type)
{
   nir_foreach_use(use, def) {
      nir_instr *use_instr = nir_src_parent_instr(use);
      if (use_instr->type != nir_instr_type_alu)
         continue;

      /* This is the ALU instruction that contains the use, not
       * nir_src_as_alu()'s ALU instruction that generated the use's def.
       */
      nir_alu_instr *use_alu = nir_instr_as_alu(use_instr);

      uint32_t i = container_of(use, nir_alu_src, src) - use_alu->src;
      assert(i < nir_op_infos[use_alu->op].num_inputs);

      if (nir_alu_type_get_base_type(nir_op_infos[use_alu->op].input_types[i]) == type)
         return true;

      if (nir_op_is_vec_or_mov(use_alu->op) && nir_def_is_used_as(&use_alu->def, type))
         return true;
   }

   return false;
}

#define INPUT_VALUE_COUNT_LOG2 3
#define INPUT_VALUE_COUNT      (1 << INPUT_VALUE_COUNT_LOG2)
#define INPUT_VALUE_MASK       ((1 << INPUT_VALUE_COUNT_LOG2) - 1)

static uint32_t
get_seed_bit_size(input_type type)
{
   if (type == BOOL)
      return 1;
   else
      return INPUT_VALUE_COUNT_LOG2;
}

/**
 * Rewrites the nir_intrinsic_unit_test_uniform_inputs to load_consts, and
 * tracks them as variables that need to be rewritten with the test inputs per
 * iteration.
 *
 * We can't emit load_consts directly in test generation, because nir_builder
 * helpers might try to fold those constants.
 */
static bool
map_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_algebraic_pattern_test *test = (nir_algebraic_pattern_test *)data;

   if (intr->intrinsic != nir_intrinsic_unit_test_uniform_input)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *load = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);

   enum input_type ty;

   if (nir_def_is_used_as(&intr->def, nir_type_bool))
      ty = BOOL;
   else if (nir_def_is_used_as(&intr->def, nir_type_float))
      ty = FLOAT;
   else if (nir_def_is_used_as(&intr->def, nir_type_uint))
      ty = UINT;
   else
      ty = INT;
   test->inputs.push_back(nir_algebraic_pattern_test_input(nir_instr_as_load_const(nir_def_instr(load)),
                                                           ty, test->fuzzing_bits));
   test->fuzzing_bits += get_seed_bit_size(ty);

   nir_def_replace(&intr->def, load);

   return true;
}

static const uint64_t uint_inputs[INPUT_VALUE_COUNT] = {
   0,
   1,
   2,
   3,
   4,
   32,
   64,
   UINT64_MAX,
};

static const int64_t int_inputs[INPUT_VALUE_COUNT] = {
   0,
   1,
   -1,
   2,
   3,
   64,
   INT64_MIN,
   INT64_MAX,
};

static const double float_inputs[INPUT_VALUE_COUNT] = {
   0,
   1,
   -1,
   0.12345,
   NAN,
   INFINITY,
   -INFINITY,
   DBL_MIN,
};

static bool
skip_test(nir_algebraic_pattern_test *test, nir_alu_instr *alu, uint32_t bit_size,
          nir_const_value tmp, int32_t src_index, bool exact)
{
   /* Always pass the test for signed zero/nan/inf sources if they are not preserved. */
   if (bit_size >= 16) {
      double val = nir_const_value_as_float(tmp, bit_size);
      if ((!exact || !(test->fp_math_ctrl & nir_fp_preserve_signed_zero)) && val == 0.0 && signbit(val)) {
         /* TODO: Could be more permissive in covering input values -- right now
          * we skip if either before or after ever consume or produce a -0.0,
          * but if the result was unchanged by the 0.0 signs of the srcs, or if
          * the two sides agreed about the sign of 0.0s produced, we could test
          * that the rest of the expression evaluated correctly.
          *
          * Also, the fp preserve flags should probably not apply to non-float
          * uses/outputs!
          */
         return true;
      }
      if ((!exact || !(test->fp_math_ctrl & nir_fp_preserve_nan)) && isnan(val))
         return true;
      if ((!exact || !(test->fp_math_ctrl & nir_fp_preserve_inf)) && isinf(val))
         return true;
   }

   switch (alu->op) {
   case nir_op_fsign:
      /* SPIRV's FSign says "If x = ±NaN, the result can be any of ±1.0 or ±0.0,
       * regardless of whether shader_float_controls is in use."
       */
      if (isnan(nir_const_value_as_float(tmp, bit_size)))
         return true;
      break;

   case nir_op_f2u8:
   case nir_op_f2u16:
   case nir_op_f2u32:
   case nir_op_f2i8:
   case nir_op_f2i16:
   case nir_op_f2i32:
      if (src_index == 0) {
         double value = nir_const_value_as_float(tmp, bit_size);
         /* Not called out in SPIRV, but a source of UB in our constant evaluation for sure. */
         if (isnan(value))
            return true;
      }
      break;

   default:
      break;
   }

   return false;
}

static bool
compare_inexact(double a, double b, uint32_t bit_size)
{
   return abs(a - b) > pow(0.5, bit_size / 4);
}

/* Returns true if this expression means the testcase passed with these input values
 * (either assert_eq was true, or we hit some UB with these inputs and the test should
 * be skipped).
 */
static bool
evaluate_expression(nir_algebraic_pattern_test *test, nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);

      if (intrinsic->intrinsic == nir_intrinsic_unit_test_assert_eq) {
         nir_const_value *src0 = tmp_value(test, intrinsic->src[0].ssa);
         nir_const_value *src1 = tmp_value(test, intrinsic->src[1].ssa);

         assert(intrinsic->src[0].ssa->bit_size == intrinsic->src[1].ssa->bit_size);
         uint32_t bit_size = intrinsic->src[0].ssa->bit_size;

         assert(intrinsic->src[0].ssa->num_components == intrinsic->src[1].ssa->num_components);
         uint32_t num_components = intrinsic->src[0].ssa->num_components;

         nir_alu_instr *alu0 = nir_src_as_alu(intrinsic->src[0]);
         nir_alu_instr *alu1 = nir_src_as_alu(intrinsic->src[1]);
         bool is_float = (alu0 && nir_alu_type_get_base_type(nir_op_infos[alu0->op].output_type) == nir_type_float) ||
                         (alu1 && nir_alu_type_get_base_type(nir_op_infos[alu1->op].output_type) == nir_type_float);

         for (uint32_t comp = 0; comp < num_components; comp++) {
            uint64_t au = nir_const_value_as_uint(src0[comp], bit_size);
            uint64_t bu = nir_const_value_as_uint(src1[comp], bit_size);

            if (au != bu) {
               if (bit_size >= 16) {
                  double af = nir_const_value_as_float(src0[comp], bit_size);
                  double bf = nir_const_value_as_float(src1[comp], bit_size);
                  if (test->exact) {
                     if (!(is_float && isnan(af) && isnan(bf)))
                        return false;
                  } else {
                     /* NOTE: we do inexact float compare even on integer
                      * outputs!  This handles (poorly) the expected inexactness
                      * of e.g. the pack_half_2x16_split ->
                      * pack_half_2x16_rtz_split transform.
                      */
                     if (compare_inexact(af, bf, bit_size))
                        return false;
                  }
               } else {
                  return false;
               }
            }
         }

         return true;
      }

      return false;
   }

   if (instr->type == nir_instr_type_load_const) {
      nir_load_const_instr *load_const = nir_instr_as_load_const(instr);

      for (uint32_t i = 0; i < load_const->def.num_components; i++)
         tmp_value(test, &load_const->def)[i] = load_const->value[i];

      return false;
   }

   nir_alu_instr *alu = nir_instr_as_alu(instr);

   uint32_t bit_size = 0;
   if (!nir_alu_type_get_type_size(nir_op_infos[alu->op].output_type))
      bit_size = alu->def.bit_size;

   nir_const_value src[NIR_ALU_MAX_INPUTS][NIR_MAX_VEC_COMPONENTS];
   for (uint32_t i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      if (bit_size == 0 &&
          !nir_alu_type_get_type_size(nir_op_infos[alu->op].input_types[i]))
         bit_size = alu->src[i].src.ssa->bit_size;

      for (uint32_t j = 0; j < nir_ssa_alu_instr_src_components(alu, i); j++) {
         nir_const_value tmp = tmp_value(test, alu->src[i].src.ssa)[alu->src[i].swizzle[j]];
         src[i][j] = tmp;

         if (skip_test(test, alu, alu->src[i].src.ssa->bit_size, tmp, i, test->exact))
            return true;
      }
   }

   if (bit_size == 0)
      bit_size = 32;

   nir_const_value *srcs[NIR_MAX_VEC_COMPONENTS];
   for (uint32_t i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
      srcs[i] = src[i];

   nir_const_value *dest = tmp_value(test, &alu->def);

   nir_component_mask_t poison;
   nir_eval_const_opcode(alu->op, dest, &poison, alu->def.num_components, bit_size, srcs, test->b->shader->info.float_controls_execution_mode);

   /* If the inputs we chose triggered UB, then skip this particular test
    * combination -- we can't assert equality of the results (and we don't have
    * the UB of NIR opcodes well enough enumerated that we can assert that UB
    * was preserved by our transformations).
    */
   if (poison)
      return true;

   for (uint32_t comp = 0; comp < alu->def.num_components; comp++) {
      if (skip_test(test, alu, bit_size, dest[comp], -1, test->exact))
         return true;
   }

   return false;
}

/** Sets the load_const values that serve as the inputs to a test iteration. */
void
nir_algebraic_pattern_test::set_inputs(uint32_t seed)
{
   for (uint32_t i = 0; i < input_count; i++) {
      nir_load_const_instr *load = inputs[i].instr;
      uint32_t seed_bit_size = get_seed_bit_size(inputs[i].ty);

      for (uint32_t comp = 0; comp < load->def.num_components; comp++) {
         uint32_t val = seed >> ((inputs[i].fuzzing_start_bit + seed_bit_size * comp) % 32);

         if (load->def.bit_size == 1) {
            load->value[comp].b = val & 1;
            continue;
         }

         /* Zero out the rest of the field. */
         load->value[comp].u64 = 0;

         switch (inputs[i].ty) {
         case BOOL:
            /* Single path here sets the bit pattern for any size of bool. */
            load->value[comp].u64 = (val & 1) ? ~0llu : 0;
            break;
         case FLOAT:
            load->value[comp] = nir_const_value_for_float(float_inputs[val & INPUT_VALUE_MASK],
                                                          load->def.bit_size);
            break;
         case UINT:
            load->value[comp] = nir_const_value_for_raw_uint(uint_inputs[val & INPUT_VALUE_MASK],
                                                             load->def.bit_size);
            break;
         case INT:
            load->value[comp] = nir_const_value_for_raw_uint(int_inputs[val & INPUT_VALUE_MASK],
                                                             load->def.bit_size);
            break;
         }
      }
   }
}

bool
nir_algebraic_pattern_test::check_variable_conds()
{
   if (variable_conds.empty())
      return true;

   nir_search_state state = {
      .range_ht = _mesa_pointer_hash_table_create(NULL),
      .numlsb_ht = _mesa_pointer_hash_table_create(NULL),
   };

   for (auto cond : variable_conds) {
      uint8_t swiz[NIR_MAX_VEC_COMPONENTS];
      int num_components = nir_ssa_alu_instr_src_components(cond.alu, cond.src_index);
      for (int i = 0; i < num_components; i++)
         swiz[i] = i;
      if (!cond.cond(&state, cond.alu, cond.src_index, num_components, swiz))
         return false;
   }

   _mesa_hash_table_fini(state.numlsb_ht, NULL);
   _mesa_hash_table_fini(state.range_ht, NULL);

   return true;
}

void
nir_algebraic_pattern_test::validate_pattern()
{
   input_count = 0;
   fuzzing_bits = 0;

   nir_function_impl *impl = nir_shader_get_entrypoint(b->shader);

   nir_validate_shader(b->shader, "validate_pattern");

   nir_shader_intrinsics_pass(b->shader, count_input, nir_metadata_all, this);
   nir_shader_intrinsics_pass(b->shader, map_input, nir_metadata_all, this);

   nir_index_ssa_defs(impl);
   tmp_values.assign(NIR_MAX_VEC_COMPONENTS * impl->ssa_alloc, nir_const_value{ 0 });

   if (expected_result != UNSUPPORTED) {
      ASSERT_EQ(expression_cond_failed, (char *)NULL);
   } else if (expression_cond_failed) {
      return;
   }

   bool result = true;

   bool overflow = fuzzing_bits > 16;
   if (overflow)
      fuzzing_bits = 16;

   nir_block *block = nir_impl_last_block(impl);

   bool all_skipped = true;
   uint32_t iterations = 1 << fuzzing_bits;
   for (uint32_t i = 0; i < iterations; i++) {
      uint32_t seed;
      if (overflow)
         seed = _mesa_hash_u32(&i);
      else
         seed = i;

      set_inputs(seed);
      if (!check_variable_conds())
         continue;

      bool passed_or_skipped = false;
      nir_foreach_instr(instr, block) {
         if (evaluate_expression(this, instr)) {
            passed_or_skipped = true;
            if (instr->type == nir_instr_type_intrinsic) {
               if (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_unit_test_assert_eq)
                  all_skipped = false;
            }
            break;
         }
      }

      if (!passed_or_skipped) {
         result = false;
         break;
      }
   }
   /* If no values produced a passing reuslt, make sure the test is marked
    * unsupported (and that nothing is marked unsupported that *was* supported).
    */
   ASSERT_EQ(all_skipped, expected_result == UNSUPPORTED);
   if (expected_result != UNSUPPORTED) {
      ASSERT_EQ(result, expected_result == PASS);
   }
}
