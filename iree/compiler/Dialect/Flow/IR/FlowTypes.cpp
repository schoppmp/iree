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

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowEnums.cpp.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

// static
DispatchTensorType DispatchTensorType::get(TensorAccess access,
                                           ArrayRef<int64_t> shape,
                                           Type elementType) {
  return Base::get(elementType.getContext(), static_cast<uint32_t>(access),
                   shape, elementType);
}

// static
DispatchTensorType DispatchTensorType::get(TensorAccess access,
                                           TensorType tensorType) {
  return DispatchTensorType::get(access, tensorType.getShape(),
                                 tensorType.getElementType());
}

TensorAccess DispatchTensorType::getAccess() const {
  return static_cast<TensorAccess>(static_cast<ImplType *>(impl)->access);
}

Type DispatchTensorType::getElementType() const {
  return static_cast<ImplType *>(impl)->elementType;
}

unsigned DispatchTensorType::getElementTypeBitWidth() const {
  return getElementType().getIntOrFloatBitWidth();
}

int64_t DispatchTensorType::getNumElements() const {
  assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
  auto shape = getShape();
  int64_t num = 1;
  for (auto dim : shape) num *= dim;
  return num;
}

int64_t DispatchTensorType::getRank() const { return getShape().size(); }

bool DispatchTensorType::hasRank() const { return true; }

int64_t DispatchTensorType::getDimSize(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return getShape()[idx];
}

bool DispatchTensorType::isDynamicDim(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return isDynamic(getShape()[idx]);
}

unsigned DispatchTensorType::getDynamicDimIndex(unsigned index) const {
  assert(index < getRank() && "invalid index");
  assert(DispatchTensorType::isDynamic(getDimSize(index)) && "invalid index");
  return llvm::count_if(getShape().take_front(index),
                        DispatchTensorType::isDynamic);
}

ArrayRef<int64_t> DispatchTensorType::getShape() const {
  return static_cast<ImplType *>(impl)->getShape();
}

int64_t DispatchTensorType::getNumDynamicDims() const {
  return llvm::count_if(getShape(), isDynamic);
}

bool DispatchTensorType::hasStaticShape() const {
  return hasRank() && llvm::none_of(getShape(), isDynamic);
}

bool DispatchTensorType::hasStaticShape(ArrayRef<int64_t> shape) const {
  return hasStaticShape() && getShape() == shape;
}

LogicalResult DispatchTensorType::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t access,
    ArrayRef<int64_t> shape, Type elementType) {
  if (!isValidElementType(elementType)) {
    return emitError() << "dispatch tensor elements must be int or float type";
  }
  if (any_of(shape, [](int64_t i) { return i < -1; })) {
    return emitError()
           << "dispatch tensor dimensions must be positive if defined";
  }
  return success();
}

template <typename T>
static T parseShapedType(DialectAsmParser &parser) {
  StringRef accessStr;
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&accessStr)) ||
      failed(parser.parseColon()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)) ||
      failed(parser.parseType(elementType)) || failed(parser.parseGreater())) {
    return {};
  }
  auto access = llvm::StringSwitch<TensorAccess>(accessStr)
                    .Case("readonly", TensorAccess::ReadOnly)
                    .Case("readwrite", TensorAccess::ReadWrite)
                    .Case("writeonly", TensorAccess::WriteOnly)
                    .Default(TensorAccess::ReadOnly);
  return T::get(access, shape, elementType);
}

static void printShapedType(DispatchTensorType &type, DialectAsmPrinter &p) {
  switch (type.getAccess()) {
    case TensorAccess::ReadOnly:
      p << "readonly";
      break;
    case TensorAccess::ReadWrite:
      p << "readwrite";
      break;
    case TensorAccess::WriteOnly:
      p << "writeonly";
      break;
    default:
      llvm_unreachable("unhandled access");
  }
  p << ":";
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      p << '?';
    } else {
      p << dim;
    }
    p << 'x';
  }
  p << type.getElementType();
}

// static
DispatchTensorType DispatchTensorType::parse(DialectAsmParser &parser) {
  return parseShapedType<DispatchTensorType>(parser);
}

void printType(DispatchTensorType &type, DialectAsmPrinter &p) {
  p << "dispatch.tensor<";
  printShapedType(type, p);
  p << '>';
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
