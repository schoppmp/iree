// Copyright 2021 Google LLC
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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

class MatmulWorkgroupTilesPadding : public OpRewritePattern<linalg::MatmulOp> {
 public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  linalg::PadTensorOp createPadTensorOp(PatternRewriter &rewriter,
                                        mlir::Location loc, Type resultType,
                                        Value input, Value padding,
                                        ArrayRef<int64_t> lowPadding,
                                        ArrayRef<int64_t> highPadding) const {
    auto padTensorOp = rewriter.create<linalg::PadTensorOp>(
        loc, resultType, input, ArrayRef<Value>{}, ArrayRef<Value>{},
        rewriter.getI64ArrayAttr(lowPadding),
        rewriter.getI64ArrayAttr(highPadding));

    int rank = padTensorOp.getResultType().getRank();
    SmallVector<Type, 4> blockArgTypes;
    blockArgTypes.assign(rank, rewriter.getIndexType());
    auto &region = padTensorOp.region();
    // `builder.createBlock` changes the insertion point within the block.
    // Create a guard to reset the insertion point of the builder after it is
    // destroyed.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&region, region.end(), blockArgTypes);
    rewriter.create<linalg::YieldOp>(loc, padding);
    return padTensorOp;
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();
    auto lhs = matmulOp.getInput(0);
    auto rhs = matmulOp.getInput(1);
    auto result = matmulOp.getOutput(0);

    if (lhs.getDefiningOp<linalg::PadTensorOp>() ||
        rhs.getDefiningOp<linalg::PadTensorOp>())
      return failure();

    auto getFullSize = [](Value operand) -> ArrayRef<int64_t> {
      auto defOp = operand.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
      if (!defOp) return {};
      TensorType shape = defOp.source()
                             .getType()
                             .cast<IREE::Flow::DispatchTensorType>()
                             .asTensorType();
      if (!shape.hasStaticShape()) return {};
      return shape.getShape();
    };

    auto lhsFullSize = getFullSize(lhs);
    auto rhsFullSize = getFullSize(rhs);

    auto workgroupTileSizes =
        getTileSizes<TilingLevel::WorkGroupTiles>(matmulOp);

    auto L1TileSizes = getTileSizes<TilingLevel::Level1Tiles>(matmulOp);
    auto vectorTileSizes = getTileSizes<TilingLevel::Level2Tiles>(matmulOp);

    llvm::outs() << "Problem size:\n";
    llvm::outs() << "lhs:" << lhsFullSize[0] << " " << lhsFullSize[1] << "\n";
    llvm::outs() << "rhs:" << rhsFullSize[0] << " " << rhsFullSize[1] << "\n";

    llvm::outs() << "workgroupTileSizes :\n"
                 << workgroupTileSizes[0] << " " << workgroupTileSizes[1]
                 << "\n";
    llvm::outs() << "vectorTileSizes:\n"
                 << vectorTileSizes[0] << " " << vectorTileSizes[1] << " "
                 << vectorTileSizes[2] << "\n";

    auto getPaddedTileSize = [](int dim, int workgoupSize,
                                int vecSize) -> SmallVector<int> {
      if (dim > workgoupSize) {
        int size = ceil((float)dim / (float)workgoupSize);
        int padValue = size - (dim / workgoupSize);
        return {workgoupSize, padValue};
      } else if (dim < vecSize) {
        int size = vecSize;
        int padValue = size - dim;
        return {vecSize, padValue};
      } else {
        int size = ceil((float)dim / (float)vecSize) * vecSize;
        int padValue = size - dim;
        return {size, padValue};
      }
    };

    auto paddedM = getPaddedTileSize(lhsFullSize[0], workgroupTileSizes[0],
                                     vectorTileSizes[0]);
    auto paddedN = getPaddedTileSize(rhsFullSize[1], workgroupTileSizes[1],
                                     vectorTileSizes[1]);

    // We don't tile K but assume it was tiled the same way as either M or N.
    auto paddedK = getPaddedTileSize(lhsFullSize[1], workgroupTileSizes[0],
                                     vectorTileSizes[2]);

    llvm::outs() << "padded sizes:" << paddedM[0] << " " << paddedN[0] << " "
                 << paddedK[0] << "\n";

    llvm::outs() << "padded paddings:" << paddedM[1] << " " << paddedN[1] << " "
                 << paddedK[1] << "\n";

    if (lhsFullSize.empty() || rhsFullSize.empty()) return failure();

    if (paddedM[1] == 0 && paddedN[1] == 0 && paddedK[1] == 0) return failure();

    auto paddingValue =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));

    auto paddedLhsType = RankedTensorType::get(
        {paddedM[0], paddedK[0]},
        lhs.getType().cast<RankedTensorType>().getElementType());

    auto paddedrhsType = RankedTensorType::get(
        {paddedK[0], paddedN[0]},
        rhs.getType().cast<RankedTensorType>().getElementType());

    auto paddedLhs =
        createPadTensorOp(rewriter, loc, paddedLhsType, lhs, paddingValue,
                          {0, 0}, {paddedM[1], paddedK[1]});

    auto paddedrhs =
        createPadTensorOp(rewriter, loc, paddedrhsType, rhs, paddingValue,
                          {0, 0}, {paddedK[1], paddedN[1]});

    auto resultType = RankedTensorType::get(
        {paddedM[0], paddedN[0]},
        result.getType().cast<RankedTensorType>().getElementType());

    auto getActualSizes = [](Value operand) {
      auto defOp = operand.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
      return defOp.sizes();
    };
    auto lhsSizes = getActualSizes(lhs);
    auto rhsSizes = getActualSizes(rhs);

    llvm::outs() << "Done: getting sizes:\n";
    llvm::outs() << "lhsSizes.size():" << lhsSizes.size() << "\n";
    llvm::outs() << "lhsSizes.size():" << rhsSizes.size() << "\n";

    SmallVector<OpFoldResult> offsets(2, rewriter.getI64IntegerAttr(0));
    SmallVector<OpFoldResult> strides(2, rewriter.getI64IntegerAttr(1));

    SmallVector<OpFoldResult> sizes = {lhsSizes[0], rhsSizes[0]};

    if (paddedM[1] == 0 && paddedN[1] == 0) {
      auto newOp = rewriter.create<linalg::MatmulOp>(
          loc, resultType, ArrayRef<Value>{paddedLhs, paddedrhs},
          ArrayRef<Value>{result});
      newOp.getOperation()->setAttr("__internal_linalg_transform__",
                                    rewriter.getStringAttr("workgroup"));
      llvm::outs() << "Getting sizes\n";
      rewriter.replaceOp(matmulOp, newOp.getResults());
    } else {
      auto staticResult = rewriter.create<linalg::InitTensorOp>(
          loc, ArrayRef<int64_t>{paddedM[0], paddedN[0]},
          matmulOp.getResults()[0]
              .getType()
              .cast<ShapedType>()
              .getElementType());
      auto newOp = rewriter.create<linalg::MatmulOp>(
          loc, resultType, ArrayRef<Value>{paddedLhs, paddedrhs},
          ArrayRef<Value>{staticResult});
      newOp.getOperation()->setAttr("__internal_linalg_transform__",
                                    rewriter.getStringAttr("workgroup"));
      llvm::outs() << "Getting sizes\n";
      // Take a subtensor of the results.
      // rewriter.create<SubTensorOp>(loc, matmulOp.getResult(0).getType(),
      // newOp.getResult(0)
      auto subtensorOp = rewriter.replaceOpWithNewOp<SubTensorOp>(
          matmulOp, newOp.getResults()[0], offsets, sizes, strides);
      llvm::outs() << "\n";
      subtensorOp->print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "return success()\n";
    return success();
  }
};

struct PadLinalgWorkgroupTilesPass
    : PassWrapper<PadLinalgWorkgroupTilesPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<MatmulWorkgroupTilesPadding>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createPadLinalgWorkgroupTilesPass() {
  return std::make_unique<PadLinalgWorkgroupTilesPass>();
}

static PassRegistration<PadLinalgWorkgroupTilesPass> pass(
    "iree-codegen-llvm-pad-linalg-workgroup-tiles",
    "Padding linalg operands on tensors.");

}  // namespace iree_compiler
}  // namespace mlir
