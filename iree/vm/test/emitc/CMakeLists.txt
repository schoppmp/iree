# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_add_all_subdirs()

if(${IREE_ENABLE_EMITC})

iree_cc_test(
  NAME
    module_test
  SRCS
    "module_test.cc"
  DEPS
    absl::strings
    iree::base::logging
    iree::base::status
    iree::testing::gtest
    iree::testing::gtest_main
    iree::vm::cc
    iree::vm::ops
    iree::vm::shims_emitc
    ::arithmetic_ops
    ::arithmetic_ops_i64
    ::assignment_ops
    ::assignment_ops_i64
    ::comparison_ops
    ::comparison_ops_i64
    ::control_flow_ops
    ::conversion_ops
    ::conversion_ops_i64
    ::global_ops
    ::list_ops
    ::shift_ops
    ::shift_ops_i64
)

iree_c_module(
  NAME
    arithmetic_ops
  SRC
    "../arithmetic_ops.mlir"
  H_FILE_OUTPUT
    "arithmetic_ops.h"
)

iree_c_module(
  NAME
    arithmetic_ops_i64
  SRC
    "../arithmetic_ops_i64.mlir"
  H_FILE_OUTPUT
    "arithmetic_ops_i64.h"
)

iree_c_module(
  NAME
    assignment_ops
  SRC
    "../assignment_ops.mlir"
  H_FILE_OUTPUT
    "assignment_ops.h"
)

iree_c_module(
  NAME
    assignment_ops_i64
  SRC
    "../assignment_ops_i64.mlir"
  H_FILE_OUTPUT
    "assignment_ops_i64.h"
)

iree_c_module(
  NAME
    comparison_ops
  SRC
    "../comparison_ops.mlir"
  H_FILE_OUTPUT
    "comparison_ops.h"
)

iree_c_module(
  NAME
    comparison_ops_i64
  SRC
    "../comparison_ops_i64.mlir"
  H_FILE_OUTPUT
    "comparison_ops_i64.h"
)

iree_c_module(
  NAME
    control_flow_ops
  SRC
    "../control_flow_ops.mlir"
  H_FILE_OUTPUT
    "control_flow_ops.h"
)

iree_c_module(
  NAME
    conversion_ops
  SRC
    "../conversion_ops.mlir"
  H_FILE_OUTPUT
    "conversion_ops.h"
)

iree_c_module(
  NAME
    conversion_ops_i64
  SRC
    "../conversion_ops_i64.mlir"
  H_FILE_OUTPUT
    "conversion_ops_i64.h"
)

iree_c_module(
  NAME
    global_ops
  SRC
    "../global_ops.mlir"
  H_FILE_OUTPUT
    "global_ops.h"
)

iree_c_module(
  NAME
    list_ops
  SRC
    "../list_ops.mlir"
  H_FILE_OUTPUT
    "list_ops.h"
)

iree_c_module(
  NAME
    shift_ops
  SRC
    "../shift_ops.mlir"
  H_FILE_OUTPUT
    "shift_ops.h"
)

iree_c_module(
  NAME
    shift_ops_i64
  SRC
    "../shift_ops_i64.mlir"
  H_FILE_OUTPUT
    "shift_ops_i64.h"
)

endif()
