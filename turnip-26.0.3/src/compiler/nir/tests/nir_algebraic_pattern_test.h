/*
 * Copyright © 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef NIR_ALGEBRAIC_PATTERN_TEST_H
#define NIR_ALGEBRAIC_PATTERN_TEST_H

#include "nir.h"
#include "nir_test.h"
#include "nir_search.h"
#include "gtest/gtest-spi.h"

class nir_algebraic_pattern_test_variable_cond {
public:
   nir_algebraic_pattern_test_variable_cond(nir_alu_instr *alu, unsigned src_index, const nir_search_variable_cond cond)
       : alu(alu), src_index(src_index), cond(cond)
   {
   }

   nir_alu_instr *alu;
   unsigned src_index;
   const nir_search_variable_cond cond;
};

enum input_type {
   BOOL,
   INT,
   UINT,
   FLOAT,
};

class nir_algebraic_pattern_test_input {
 public:
   nir_algebraic_pattern_test_input(nir_load_const_instr *instr, input_type ty, uint32_t fuzzing_start_bit)
       : instr(instr), ty(ty), fuzzing_start_bit(fuzzing_start_bit)
   {
   }

   nir_load_const_instr *instr;
   input_type ty;
   uint32_t fuzzing_start_bit;
};

enum expected_result {
   PASS,
   XFAIL,
   UNSUPPORTED,
};

class nir_algebraic_pattern_test : public nir_test {
 protected:
   nir_algebraic_pattern_test(const char *name);
   virtual ~nir_algebraic_pattern_test();

   void set_inputs(uint32_t seed);
   bool check_variable_conds();
   void validate_pattern();

 public:
   uint32_t input_count;
   std::vector<nir_algebraic_pattern_test_input> inputs;
   uint32_t fuzzing_bits;
   bool exact = true;
   enum expected_result expected_result = PASS;
   const char *expression_cond_failed = NULL;
   nir_fp_math_control fp_math_ctrl = (nir_fp_math_control)(nir_fp_preserve_signed_zero |
                                                            nir_fp_preserve_inf |
                                                            nir_fp_preserve_nan);
   std::vector<nir_algebraic_pattern_test_variable_cond> variable_conds;
   std::vector<nir_const_value> tmp_values;
};

#endif
