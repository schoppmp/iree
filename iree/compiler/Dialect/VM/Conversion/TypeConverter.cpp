// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-vm"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

TypeConverter::TypeConverter(TargetOptions targetOptions)
    : targetOptions_(targetOptions) {
  // Variant means opaque in VM.
  addConversion([](IREE::VariantType type) {
    return IREE::VM::OpaqueType::get(type.getContext());
  });

  // All ref types are passed through unmodified.
  addConversion([](IREE::VM::RefType type) { return type; });

  // Wrap ref types.
  addConversion([](Type type) -> Optional<Type> {
    if (RefType::isCompatible(type)) {
      return RefType::get(type);
    }
    return llvm::None;
  });

  // Pointer types remain as pointer types types are passed through unmodified.
  addConversion([this](IREE::PtrType type) -> Optional<Type> {
    // Recursively handle pointer target types (we want to convert ptr<index> to
    // ptr<i32>, for example).
    auto targetType = convertType(type.getTargetType());
    if (!targetType) {
      return llvm::None;
    }
    return IREE::PtrType::get(targetType);
  });

  // Convert integer types.
  addConversion([this](IntegerType integerType) -> Optional<Type> {
    if (integerType.isInteger(32)) {
      // i32 is always supported by the runtime.
      return integerType;
    } else if (integerType.getIntOrFloatBitWidth() < 32) {
      // Promote i1/i8/i16 -> i32.
      return IntegerType::get(integerType.getContext(), 32);
    } else if (integerType.isInteger(64)) {
      if (targetOptions_.i64Extension) {
        // i64 is supported by the VM, use directly.
        return integerType;
      } else if (targetOptions_.truncateUnsupportedIntegers) {
        // i64 is not supported and we still want to compile, so truncate to i32
        // (unsafe if all bits are actually required!).
        return IntegerType::get(integerType.getContext(), 32);
      }
    }
    return llvm::None;
  });

  // Convert floating-point types.
  addConversion([this](FloatType floatType) -> Optional<Type> {
    if (floatType.getIntOrFloatBitWidth() < 32) {
      if (targetOptions_.f32Extension) {
        // Promote f16 -> f32.
        return FloatType::getF32(floatType.getContext());
      } else {
        // f32 is not supported; can't compile.
        return llvm::None;
      }
    } else if (floatType.isF32()) {
      if (targetOptions_.f32Extension) {
        return floatType;
      } else {
        // f32 is not supported; can't compile.
        return llvm::None;
      }
    } else if (floatType.isF64()) {
      if (targetOptions_.f64Extension) {
        // f64 is supported by the VM, use directly.
        return floatType;
      } else if (targetOptions_.f32Extension &&
                 targetOptions_.truncateUnsupportedFloats) {
        // f64 is not supported and we still want to compile, so truncate to
        // f32 (unsafe if all bits are actually required!).
        return FloatType::getF32(floatType.getContext());
      }
    }
    return llvm::None;
  });

  // Convert index types to the target bit width.
  addConversion([this](IndexType indexType) -> Optional<Type> {
    return IntegerType::get(indexType.getContext(), targetOptions_.indexBits);
  });

  // Vectors are used for arbitrary byte storage.
  addConversion([](VectorType vectorType) -> Optional<Type> {
    return IREE::VM::RefType::get(
        IREE::VM::BufferType::get(vectorType.getContext()));
  });

  // Convert ranked shape types (expanding all dims).
  addConversion([this](Shape::RankedShapeType rankedShape,
                       SmallVectorImpl<Type> &results) {
    auto indexType =
        IntegerType::get(rankedShape.getContext(), targetOptions_.indexBits);
    for (int i = 0; i < rankedShape.getRank(); ++i) {
      if (rankedShape.isDimDynamic(i)) {
        results.push_back(indexType);
      }
    }
    return success();
  });

  // TODO(b/145876978): materialize conversion for other types
  addArgumentMaterialization([](OpBuilder &builder,
                                Shape::RankedShapeType resultType,
                                ValueRange inputs, Location loc) -> Value {
    LLVM_DEBUG(llvm::dbgs()
               << "MATERIALIZE CONVERSION: " << resultType << "\n");
    return builder.create<Shape::MakeRankedShapeOp>(loc, resultType, inputs);
  });

  addSourceMaterialization([](OpBuilder &builder, IndexType type,
                              ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1 || !inputs.front().getType().isa<IntegerType>()) {
      return nullptr;
    }
    return builder.create<IndexCastOp>(loc, type, inputs.front());
  });

  addTargetMaterialization(
      [](OpBuilder &builder, IntegerType type, ValueRange inputs,
         Location loc) -> Value { return inputs.front(); });
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
