---
title: "01 | Accustoming Yourself to C++"
date: 2024-06-29T00:01:00+08:00
lastmod: 2024-06-29T15:00:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
categories:
    - learn
tags:
    - c/c++
description: My learning notes of "Effective C++" by Scott Meyers.
summary: My learning notes of "Effective C++" by Scott Meyers. 
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---


## 01 View C++ as a federation of languages.

C++是多重范型编程语言 (multiparadigm programming language) ,  
同时支持过程形式 (procedural) , 面向对象形式 (object-oriented) , 函数形式 (functional) , 泛型形式 (generic) , 元编程形式 (metaprogramming) 的语言.  

## 02 Prefer consts, enums, inlines to #defines
### a. 指针常量
当希望在 header files 内定义一个指向常量的常量指针, 以 char*-based 字符串为例, 必须写 const 两次:

```cpp
// File header.h
const char* const authorName = "Scott Meyers";
```

**[note]** const 全局对象写入头文件并被不同 cpp 文件包含时, 不会出现 redefination 错误.

### b. class 的静态常量成员

当要声明一个 class 专属的常量, 则需要让它成为 class 的一个成员; 为确保此常量至多只有一份实体, 必须让它成为 static 成员:

```cpp
// File GamePlayer.h
class GamePlayer {
private:
    static const int NumTurns = 5;  // Declaration of a const
    int scores[NumTurns];
}
```

**以上语句是声明式而非定义式.**  

通常 C++ 要求你对使用的任何东西提供一个定义式,  
但是如果你使用的是一个 class 的专属 static 常量, 并且还是一个整数类型 (intergral type, 例如 int, char, bool), 则可以只声明就使用.  
但是你如果想**对 class 的专属整数类型 static 常量取地址**, 就必须对该常量提供一个额外的定义式:

```cpp
// File GamePlayer.cpp
const int GamePlayer::NumTurns;  // Definition of a const
```

以上代码应写入一个 cpp 文件而非头文件.  
由于 class 常量在声明时已经获得初始值, 因此定义时可以不再设置初始值.

## c. enum hack

现代编译器都支持 static 成员在声明式上获得初始值, 但是旧的编译器可能不支持.  
对于这种情况, 可以将初始值写在类外对类的专属常量的定义中.  

```cpp
// File CostEstimate.h
class A {
private:
    static const double FudgeFactor;  // Declaration of a class static const
}

// File CostEstimate.cpp
const double CostEstimate::FudgeFactor = 1.35;  // Defination of a class static const
```

采用这种写法, 如果在 class 编译期间需要一个常量值 (例如上述 `GamePlayer::scores` 的数组声明式中, 数组的大小必须在编译期间确定) , 一些编译器可能会 (错误地) 报出问题.   

一种补偿做法是采用 **enum hack** 技术.

enum hack 技术的理论基础为: 一个属于枚举类型 (enumerated type) 的数值可权充 int 被使用.

```cpp
class GamePlayer2 {
private:
    enum { NumTurns = 5};  // "the enum hack" - let {NumTurns} be a marker of 5
    int scores[NumTurns];  // valid
}

```

enum hack 的行为与 #define 较像 (而不是 const) .  
例如取一个 const 对象的地址是合法的, 但是取一个 enum 对象的地址不合法; 取一个 #define 对象的地址通常也不合法.  
优秀的编译器不会为 "整数型 const 对象" 设定额外的储存空间, 但是不够优秀的编译器可能会; enum 和 #define 一定不会导致非必要的内存分配.  

enum hack 是 template metaprograming 的基础技术.

## d. inline
尽可能利用 template inline functions 代替 #define 实现的 macros.

```cpp
#define CALL_WITH_MAX(a, b) f((a)>(b)?(a):(b))

int a = 5, b = 0;
CALL_WITH_MAX(++a, b);     // a is incremented twice
CALL_WITH_MAX(++a, b+10);  // a is incremented once

template<class T>
inline void callWithMax(const T& a, const T& b) {
    f(a > b ? a : b);
}
```

# 03 Use const whenever possible

令函数返回一个常量值, 往往可以降低因客户错误而造成的例外.  例如:

```cpp
class Rational {...}
const Rational operator* (const Rational& lhs, const Rational& rhs) {...}

if((a * b) = c)  // Should be (a * b) == c
    ...
```

因为返回值是 const , 编译器会直接提示错误. 如果不是 const, 有可能隐式地对 bool 进行转换.

### 成员函数如何是 const 意味着什么?

有两个流行概念: **bitwise constness** (aka. **physical constness**) 和 **logical constness**.

**bitwise const** 阵营的人相信, 成员函数只有在不更改类内任何成员变量(static 变量除外)时才可以说是 const (i.e., 不更改类内任何一个 bit).  
这种论点的好处在于很容易侦测违反点: 编译器只需寻找成员变量的赋值动作即可.

然而, 存在成员函数虽然不十足具备 const 性质, 但能通过 bitwise 测试. 参考以下例子:

```cpp
class CTextBlock {
public:
    ...
    char& operator[](std::size_t position) const {
        return pText[position];  // Not suitable
    }
private:
    char* pText;
}
```

注意, 以上写法并不合适 (但能通过编译器的测试), 应该返回 `const char&` .

**logical constness** 阵营的人相信, 一个 const 成员函数可以修改所改对象内的某些 bits, 但只有在客户侦测不出的情况下才能如此. 参考以下代码:

```cpp
class CTextBlock {
public:
    std::size_t length() const {
        if(!lengthIsValid) {
            textLength = std::strlen(pText);
            lengthIsValid = true;
        }
        return textLength;
    }
private:
    char* pText;
    mutable std::size_t textLength;
    mutable bool lengthIsValid;
}
```

### 避免 const 和 non-const 成员函数的重复

以下情况可以这样写:

```cpp
class TextBlock {
public:
    const char& operator[](std::size_t position) const {
        return text[position];
    }
    char& operator[](std::size_t position) {
        return 
            const_cast<char&>(
                static_case<const TextBlock&>(*this)[position]
            );
    }
}
```

注意不要令 const 版本调用 non-const 版本.  
const 成员函数承诺绝不改变其对象的逻辑状态, 而 non-const 函数没有.  
若要令这样的代码通过编译, 你必须使用一个 const-cast 将 \*this 身上的 const 性质解放, 这是乌云罩顶的清晰前兆.

## 04 Make Sure the Objects are initialized before they are used

### a. Member initialization list
总是使用成员初始值列表.

C++有非常固定的 "成员初始化顺序" : base classes  更早于 derived classes 被初始化, 而 class 的成员变量总是以其声明的次序被初始化.
当在成员初始值列中列出各个成员时, 最好**总是以其声明次序为次序**.

### b. Initialize static objects

local static 对象指在函数内部定义的 static 对象; 其余 static 对象均为 non-local static 对象.  
static 对象的寿命为从 "被构造出来" 到 "函数结束".

不同编译单元中的 non-local static 对象由 imlicit template instantiation 形成, 不可能推断出正确的初始化次序.  
因此若 A 是一个 non-local static 对象, 另一个 non-local static 对象 B 需要依赖已经初始化的 A 来初始化, 这时很可能出现错误.

一种手法可以避免这种错误: 将每个 non-local static 对象搬到自己的专属函数内, 这些函数返回一个 reference 指向所含的对象 (i.e., 用 local static 对象替换 non-local static 对象). 参考以下代码:

```cpp
class FileSystem {...};
inline FileSystem& tfs() {
    static FileSystem fs;
    return fs;
}

class Directory {...};
Directory::Directory() {
    ...
    std::size_t disks = tfs().numDisks();
    ...
}
inline Directory& tempDir() {
    static Directory td;
    return td;
}
```

只有当函数 `tempDir()` 首次被调用时, static 对象 `td` 才会被创建(初始化);  
而 `td` 初始化时调用了函数 `tfs()` , 此时一定会创建对象 `fs` .

这种手法的理论基础是: **C++ 保证, 函数内的 local static 对象会在 "该函数被调用期间" "首次遇上该对象的定义式" 时被初始化**.

由于以上例子中的函数 "内含 static 对象" , 在多线程系统中它们带有不确定性.  
注意: 任何一种 non-const static 对象 (不论是 local 还是 non-local) , 在多线程环境下 "等待某事发生" 都会出现麻烦.  
处理这个麻烦的一种做法是: 在程序的单线程启动阶段 (single-threaded startup portion) 手工调用所有 reference-returning 函数, 以便消除和初始化有关的 "竞速形式 (race conditions)" .
