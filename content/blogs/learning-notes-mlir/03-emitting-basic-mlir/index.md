---
title: "03 | Emitting Basic MLIR"
date: 2024-07-30T14:45:00+08:00
lastmod: 2024-07-30T15:54:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
categories:
    - notes
tags:
    - mlir
description: My learning notes of MLIR.
summary: My learning notes of MLIR.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Run Example

```bash
export LLVM_PROJ_HOME="/path/to/llvm-project"
export MLIR_HOME="$LLVM_PROJ_HOME/mlir"
export TOY_CH2_HOME="$MLIR_HOME/examples/toy/Ch2"
```

Add built binary to `PATH`:

```bash
export PATH="/path/to/llvm-project/build/bin:$PATH"
```

Create a new file `$MLIR_HOME/input.toy` ; Add the following content to the file:

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

Generate MLIR (Multi-Level Intermediate Representation):

```bash
toyc-ch2 $TOY_CH2_HOME/input.toy -emit=mlir
```

Generate AST (Abstract Syntax Tree):

```bash
toy-ch2 $TOY_CH2_HOME/input.toy -emit=ast
```

## 2. Add an Operator
### 2.1. Define the Operation

Add following code to `$TOY_CH2_HOME/include/toy/Ops.td`:

```tablegen
// SubtractOp

def SubtractOp : Toy_Op<"subtract"> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "subtract" operation performs element-wise subtraction between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

Build the MLIR with the script provided in [02 | Setupt the Environment of MLIR]():

```bash
$LLVM_PROJ_HOME/scripts/build-mlir.sh
```

Build errors pop out, because:

- `hasCustomAssemblyFormat` is assigned with `1`, but the custom parser and printer method is not implemented.
- `OpBuilder` is not implemented.

Dpn't be worried. These errors will be handled later. Instead, note that the C++ implementation of class `SubtractOp` has been generated in `$LLVM_PROJ_HOME/build/tools/mlir/examples/toy/Ch2/include/toy/Ops.h.inc`, and as a result, you are now able to use the `SubtractOp` in other source files.

### 2.2. Implement the Operations

To implement custom parser and printer methods as well as `OpBuilder`, add the following code to `$TOY_CH2_HOME/mlir/Dialect.cpp`:

```cpp
// SubtractOp

void SubtractOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult SubtractOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void SubtractOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }
```

### 2.3. Emit `-` Operator

Go to `$TOY_CH2_HOME/mlir/MLIRGen.cpp`, locate function `mlirGen` and add the specific case for `-`, as shown below:

```cpp
mlir::Value mlirGen(BinaryExprAST &binop) {
    // ...
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    case '-':
      return builder.create<SubtractOp>(location, lhs, rhs);
    }
    // ...
}
```

Rebuild the MLIR with the script provided in [02 | Setupt the Environment of MLIR]():

```bash
$LLVM_PROJ_HOME/scripts/build-mlir.sh
```

### 2.4. Test the `-` Operator

Change the content of `$MLIR_HOME/input.toy` to:

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  var e = a - b;
  print(e);
}
```

Generate MLIR:

```bash
toyc-ch2 $TOY_CH2_HOME/input.toy -emit=mlir
```

