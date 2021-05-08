
// Copyright 2020 Google LLC
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

#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

// TODO(ravishankarm): This needs to be put in a common place for the CPU and
// GPU backends to use.
static llvm::cl::list<unsigned> clLLVMTileSizes(
    "iree-llvm-tile-size",
    llvm::cl::desc("Set tile sizes to use for tiling Linalg operations in "
                   "LLVM code generation"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

static llvm::cl::opt<int> matmulWorkgroupTileSize(
    "iree-codegen-llvm-matmul-workgroup-size",
    llvm::cl::desc(
        "linalg.matmul tile size for workgroups spliting of M, N dimension"),
    llvm::cl::init(64));
static llvm::cl::opt<int> matmulL1TileSize(
    "iree-codegen-llvm-matmul-l1-size",
    llvm::cl::desc(
        "linalg.matmul tile size for L1 spliting of M, N, K dimension"),
    llvm::cl::init(32));
static llvm::cl::opt<int> matmulVectorSize(
    "iree-codegen-llvm-matmul-vector-size",
    llvm::cl::desc("linalg.matmul vector tile size"), llvm::cl::init(4));

static llvm::cl::opt<int> batchMatmulWorkgroupTileSize(
    "iree-codegen-llvm-batch-matmul-workgroup-size",
    llvm::cl::desc("linalg.batch_matmul tile size for workgroups spliting of "
                   "M, N dimension"),
    llvm::cl::init(32));
static llvm::cl::opt<int> batchMatmulL1TileSize(
    "iree-codegen-llvm-batch-matmul-l1-size",
    llvm::cl::desc(
        "linalg.batch_matmul tile size for L1 spliting of M, N, K dimensions"),
    llvm::cl::init(16));
static llvm::cl::opt<int> batchMatmulL2TileSize(
    "iree-codegen-llvm-batch-matmul-vector-size",
    llvm::cl::desc("linalg.batch_matmul vector tile size"), llvm::cl::init(4));

static llvm::cl::opt<int> genericOpsWorkgroupTileSize(
    "iree-codegen-llvm-generic-ops-workgroup-size",
    llvm::cl::desc(
        "linalg.generic and linalg.indexed_generic workgroup tile size"),
    llvm::cl::init(128));

template <TilingLevel tilingLevel>
llvm::SmallVector<int64_t, 4> getTileSizes(Operation *op) {
  if (auto contractionOp = dyn_cast<linalg::ContractionOpInterface>(op)) {
    if (contractionOp.isRowMajorMatmul()) {
      int mWorkgroupSize = matmulWorkgroupTileSize;
      int nWorkgroupSize = matmulWorkgroupTileSize;
      int mL1TileSize = matmulL1TileSize;
      int nL1TileSize = matmulL1TileSize;
      int kL1TileSize = matmulL1TileSize;
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
        // Returns the original problem size before tiling.
        auto getOriginalOperandShape = [](Value operand) {
          if (auto dispatchLoadOp =
                  operand.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
            return dispatchLoadOp.source()
                .getType()
                .cast<IREE::Flow::DispatchTensorType>()
                .getShape();
          }
          if (auto operandParent = operand.getDefiningOp<memref::SubViewOp>()) {
            return operandParent.source()
                .getType()
                .cast<ShapedType>()
                .getShape();
          }
          if (auto operandParent = operand.getDefiningOp<SubTensorOp>()) {
            return operandParent.source()
                .getType()
                .cast<ShapedType>()
                .getShape();
          }
          if (auto operandParent = operand.getDefiningOp<memref::AllocaOp>()) {
            return operandParent.getType().cast<ShapedType>().getShape();
          }
          return ArrayRef<int64_t>{};
        };

        auto lhsShape = getOriginalOperandShape(matmulOp.inputs()[0]);
        auto rhsShape = getOriginalOperandShape(matmulOp.inputs()[1]);

        if (!lhsShape.empty() && !rhsShape.empty()) {
          // Find largest tile size that is a multiple of the vector size.
          auto getTileSize = [](int dim, int maxSize) {
            if (dim < matmulVectorSize) return matmulVectorSize.getValue();
            for (int i = std::min(maxSize, dim); i > 0; --i) {
              if (dim % i == 0 && i % matmulVectorSize == 0) {
                return i;
              }
            }
            return maxSize;
          };
          mWorkgroupSize = getTileSize(lhsShape[0], mWorkgroupSize);
          nWorkgroupSize = getTileSize(rhsShape[1], nWorkgroupSize);
          mL1TileSize = getTileSize(mWorkgroupSize, mL1TileSize);
          nL1TileSize = getTileSize(nWorkgroupSize, nL1TileSize);
          kL1TileSize = getTileSize(rhsShape[0], kL1TileSize);
        }
      }

      switch (tilingLevel) {
        case TilingLevel::WorkGroupTiles: {
          return {mWorkgroupSize, nWorkgroupSize};
        }
        case TilingLevel::Level1Tiles: {
          return {mL1TileSize, nL1TileSize, kL1TileSize};
        }
        case TilingLevel::Level2Tiles: {
          return {matmulVectorSize, matmulVectorSize, matmulVectorSize};
        }
      }
    }
    if (contractionOp.isRowMajorBatchMatmul()) {
      switch (tilingLevel) {
        case TilingLevel::WorkGroupTiles: {
          return {1, batchMatmulWorkgroupTileSize,
                  batchMatmulWorkgroupTileSize};
        }
        case TilingLevel::Level1Tiles: {
          return {1, batchMatmulL1TileSize, batchMatmulL1TileSize,
                  batchMatmulL1TileSize};
        }
        case TilingLevel::Level2Tiles: {
          return {1, batchMatmulL2TileSize, batchMatmulL2TileSize,
                  batchMatmulL2TileSize};
        }
      }
    }
  }

  if (isa<linalg::GenericOp>(op)) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    switch (tilingLevel) {
      case TilingLevel::WorkGroupTiles: {
        llvm::SmallVector<int64_t, 4> workgroupTileSizes;
        int iterationRank = linalgOp.iterator_types().size();
        for (int i = 0; i < std::min(iterationRank, 3); ++i) {
          auto iteratorType = linalgOp.iterator_types()[i];
          if (iteratorType.cast<StringAttr>().getValue() ==
              getParallelIteratorTypeName()) {
            workgroupTileSizes.push_back(genericOpsWorkgroupTileSize);
          } else {
            // Don't tile workgroup across reduction dimensions.
            workgroupTileSizes.push_back(0);
          }
        }
        return workgroupTileSizes;
      }
      // TODO(ataei): Set the parameters when we enable vectorization.
      case TilingLevel::Level1Tiles:
      case TilingLevel::Level2Tiles:
        return {1, 1, 1};
    }
  }

  return {1, 1, 1};
}

#define DEFINE_TILE_SIZE_FN(tilingLevel)                                      \
  template <>                                                                 \
  SmallVector<Value, 4> TileSizeFn::get<tilingLevel>(OpBuilder & builder,     \
                                                     Operation * operation) { \
    auto tileSizes = getTileSizes<tilingLevel>(operation);                    \
    if (tileSizes.empty()) return {};                                         \
    SmallVector<Value, 4> tileSizesVal;                                       \
    tileSizesVal.reserve(tileSizes.size());                                   \
    for (auto val : tileSizes) {                                              \
      tileSizesVal.push_back(                                                 \
          builder.create<ConstantIndexOp>(operation->getLoc(), val));         \
    }                                                                         \
    return tileSizesVal;                                                      \
  }

DEFINE_TILE_SIZE_FN(TilingLevel::WorkGroupTiles)
DEFINE_TILE_SIZE_FN(TilingLevel::Level1Tiles)
DEFINE_TILE_SIZE_FN(TilingLevel::Level2Tiles)

#undef DEFINE_TILE_SIZE_FN

bool isDispatchOp(Operation *op) {
  if (auto contractionOp = dyn_cast<linalg::ContractionOpInterface>(op)) {
    if (contractionOp.isRowMajorMatmul() ||
        contractionOp.isRowMajorBatchMatmul()) {
      return true;
    }
  }
  if (isa<linalg::GenericOp>(op)) return true;
  return false;
}

Optional<LaunchConfig> initCPULaunchConfig(
    MLIRContext *context, const linalg::LinalgDependenceGraph &dependenceGraph,
    ArrayRef<linalg::LinalgOp> linalgOps) {
  LaunchConfig config;

  Optional<linalg::LinalgOp> rootOperation = llvm::None;
  for (auto linalgOp : linalgOps) {
    if (!isDispatchOp(linalgOp)) continue;
    if (rootOperation) {
      linalgOp.emitError(
          "unhandled multiple root operations in dispatch region");
      return llvm::None;
    }
    rootOperation = linalgOp;
    SmallVector<int64_t, 4> opTileSizes;
    if (!clLLVMTileSizes.empty()) {
      opTileSizes.assign(clLLVMTileSizes.begin(), clLLVMTileSizes.end());
    } else {
      opTileSizes = getTileSizes<TilingLevel::WorkGroupTiles>(linalgOp);
    }
    config.setTileSizes(linalgOp, opTileSizes, 0);
    config.setRootOperation(linalgOp);
  }
  if (!rootOperation) {
    return config;
  }
  if (failed(propogateRootOperationLaunchConfig(config, *rootOperation,
                                                dependenceGraph)))
    return llvm::None;
  return config;
}

}  // namespace iree_compiler
}  // namespace mlir
