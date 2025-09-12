---
title: "Toy Ch4 | Enabling Generic Transformation with Interfaces"
date: 2024-08-01T11:11:11+08:00
lastmod: 2024-08-01T15:06:11+08:00
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

> Reference: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/  

**Note**: Check [Setup the Environment of MLIR](../setup-the-environment-of-mlir/) for the environment setup.


## 1. Introduction

**Problem**: Naively implementing each transformation for each dialect leads to large amounts of code duplication, as the internal algorithms are generally very similar.

**Solution**: To provide the ability for transformations to opaquely hook into dialects like Toy to get the information they need.

## 2. Add OpPrintInterface

Define a new env var:

```bash
export TOY_CH4_HOME="$MLIR_HOME/examples/toy/Ch4"
```

### 2.1. Define OpPrintInterface

First, create a new file `$TOY_CH4_HOME/include/toy/OpPrintInterface.td`, define `OpPrintOpInterface` with method `opPrint` which returns a `std::string`:

```tablegen
#ifndef PRINT_INTERFACE
#define PRINT_INTERFACE

include "mlir/IR/OpBase.td"

def OpPrintOpInterface : OpInterface<"OpPrint">
{
    let description = [{
        Interface to print something in an operator.
    }];

    let methods = [
        InterfaceMethod< "Print some information in the current operation", "std::string", "opPrint" >
    ];
}


#endif // PRINT_INTERFACE
```

Second, create a file `$TOY_CH4_HOME/include/toy/OpPrintInterface.hpp`:

```cpp
#ifndef OPPRINTINTERFACE_HPP_
#define OPPRINTINTERFACE_HPP_

#include "mlir/IR/OpDefinition.h"

namespace mlir::toy {

/// Include the auto-generated declarations.
#include "toy/OpPrintOpInterface.h.inc"

} // namespace mlir::toy


#endif 
```

Third, in `$TOY_CH4_HOME/include/toy/Dialect.h`, include the new interface:

```cpp
#include "toy/OpPrintInterface.hpp"
```

Fourth, make some modifications in `$TOY_CH4_HOME/include/toy/Ops.td`. Include the interface's td at the beginning of the file:

```td
include "toy/OpPrintInterface.td"
```

Then, for example, change the `AddOp` to declare that it implements the `OpPrint` interface:

```tablegen
def AddOp : Toy_Op<"add", [
    Pure, 
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<OpPrintOpInterface>
  ]> {
  // ...
}
```

> You can also do the similar declaration for other operations.

Finally, some CMakeLists need to be modified. 

Add following lines in `$TOY_CH4_HOME/include/toy/CMakeLists.txt`:

```cmake
set(LLVM_TARGET_DEFINITIONS OpPrintInterface.td)
mlir_tablegen(OpPrintOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(OpPrintOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh4OpPrintInterfaceIncGen)
```

Change the `add_toy_chapter` in `$TOY_CH4_HOME/CMakelists.txt` to:

```cmake
add_toy_chapter(toyc-ch4
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ShapeInferencePass.cpp
  mlir/OpPrintInterfacePass.cpp
  mlir/ToyCombine.cpp

  DEPENDS
  ToyCh4OpsIncGen
  ToyCh4ShapeInferenceInterfaceIncGen
  ToyCh4CombineIncGen
  ToyCh4OpPrintInterfaceIncGen
  )
```

To match the listed source files, create a blank file `$TOY_CH4_HOME/mlir/OpPrintInterfacePass.cpp`. We will implement the pass later.

Now build the MLIR:

```bash
bash $LLVM_PROJ_HOME/scripts/build-mlir.sh
```

Errors pop out because we haven't implemented the `OpPrintInterface` which is declared in `AddOp`. Don't worry, it will be implemented in the next section.
 
Now you can check the generated C++ class declarations in, for example,  `$LLVM_PROJ_HOME/build/tools/mlir/examples/toy/Ch4/include/toy/OpPrintOpInterface.h.inc`.

### 2.3. Implement OpPrintInterface

Since the interface is declared in `AddOp` (which is actually implemented by inheriting `OpPrintOpInterface` which provides a pure virtual function `opPrint`), we need to implement the function.

In `$TOY_CH4_HOME/mlir/Dialect.cpp`, add the following code:

```cpp
std::string AddOp::opPrint() { return "I am AddOp"; }
```

### 2.4. Implement OpPrintPass

In `$TOY_CH4_HOME/include/toy/Passes.h`, add the following line under `mlir/examples/toy/Ch4/include/toy/Passes.h` (inside namespace `mlir::toy`):

```cpp
std::unique_ptr<Pass> createOpPrintPass();
```

In `$TOY_CH4_HOME/mlir/OpPrintInterfacePass.cpp`, add the following code:

```cpp
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/OpPrintInterface.hpp"
#include "toy/Passes.h"
#include <iostream>
#include <memory>

using namespace mlir;
using namespace toy;

#include "toy/OpPrintOpInterface.cpp.inc"

namespace
{
struct OpPrintIntercfacePass
    : public mlir::PassWrapper<OpPrintIntercfacePass, OperationPass<toy::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpPrintIntercfacePass)

    void runOnOperation() override
    {
        auto f = getOperation();
        f.walk([&](mlir::Operation* op) {
            if (auto shapeOp = dyn_cast<OpPrint>(op)) {
                std::cout << shapeOp.opPrint() << std::endl;
            }
        });
    }
};

}  // namespace

std::unique_ptr<Pass> createOpPrintPass()
{
    return std::make_unique<OpPrintIntercfacePass>();
}
```

### 2.5. Add Pass to Pass Manager

In `$TOY_CH4_HOME/toyc.cpp`, add the following line after `mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();`:

```cpp
optPM.addPass(mlir::toy::createOpPrintPass());
```

Implement `OpPrintPass` in `$TOY_CH4_HOME/mlir/OpPrintInterfacePass.cpp`:

```cpp
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/OpPrintInterface.hpp"
#include "toy/Passes.h"
#include <iostream>
#include <memory>

using namespace mlir;
using namespace toy;

#include "toy/OpPrintOpInterface.cpp.inc"

namespace
{
struct OpPrintIntercfacePass
    : public mlir::PassWrapper<OpPrintIntercfacePass, OperationPass<toy::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpPrintIntercfacePass)

    void runOnOperation() override
    {
        auto f = getOperation();
        f.walk([&](mlir::Operation* op) {
            if (auto shapeOp = dyn_cast<OpPrint>(op)) {
                std::cout << shapeOp.opPrint() << std::endl;
            }
        });
    }
};

}  // namespace

std::unique_ptr<Pass> mlir::toy::createOpPrintPass()
{
    return std::make_unique<OpPrintIntercfacePass>();
}
```

### 2.6. Test the Pass

Now, rebuild the MLIR:

```bash
bash $LLVM_PROJ_HOME/scripts/build-mlir.sh
```

Run the Test:

```bash
toyc-ch4 $MLIR_HOME/test/Examples/Toy/Ch4/codegen.toy -emit=mlir -opt
```