/*
 * Copyright © 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "brw_builder.h"

class lower_regioning_test : public brw_shader_pass_test {
};

TEST_F(lower_regioning_test, sel_ud_d_d)
{
   brw_builder bld = make_shader();

   brw_reg dest = vgrf(bld, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, BRW_TYPE_D);
   brw_reg src1 = vgrf(bld, BRW_TYPE_D);

   bld.SEL(dest, src0, src1)
      ->saturate = true;

   EXPECT_NO_PROGRESS(brw_lower_regioning, bld);
}
