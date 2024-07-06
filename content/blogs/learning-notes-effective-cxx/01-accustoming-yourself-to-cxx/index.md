---
title: "01 | Accustoming Yourself to C++"
date: 2024-06-29T00:01:00+08:00
lastmod: 2024-06-29T15:00:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
categories:
    - notes
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


## Term 01: View C++ as a federation of languages.

Today's C++ is a multiparadigm programming language, one supporting a combination of procedural, object-oriented, functional, generic, and metaprogramming features. 


## Term 02: Prefer consts, enums, inlines to #defines

The substitution of a macro could result in multiple copies of the object in your object code, while the use of the constant should never result in more than one copy.

### 🎼 Constant Pointer

To define a constant char*-based string in a header file, for example, you have to write const twice:

```cpp
// File header.h
const char* const authorName = "Scott Meyers";
```

> 💡 **Note**: A constant object can be defined in a header file, and there will be no redefinition error when the header file is included in multiple source files.

### 🎼 Static Constant Members of a Class

To limit the scope of a constant to a class, you must make it a member, and to ensure there's at most one copy of the constant, you must make it a static member:

```cpp
// File GamePlayer.h
class GamePlayer {
private:
    static const int NumTurns = 5;  // Declaration of a const
    int scores[NumTurns];
}
```

What you see above is a declaration for `NumTurns`, not a definition. 

Usually, C++ requires that you provide a definition for anything you use, but classspecific constants that are **static and of integral type** (e.g., integers, chars, bools) are an exception. 

As long as you don't take their address, you can declare them and use them without providing a definition. If you do take the address of a class constant, or if your compiler incorrectly insists on a definition even if you don't take the address, you should provide a separate definition like this:

```cpp
// File GamePlayer.cpp
const int GamePlayer::NumTurns;  // Definition of a const
```

You put this in an implementation file, not a header file. Because the initial value of class constants is provided where the constant is declared (e.g., `NumTurns` is initialized to 5 when it is declared), no initial value is permitted at the point of definition.


For non-integral types, you must provide a definition for the constant in the header file, like this:

```cpp
// File CostEstimate.h
class A {
private:
    static const double FudgeFactor;  // Declaration of a class static const
}

// File CostEstimate.cpp
const double CostEstimate::FudgeFactor = 1.35;  // Defination of a class static const
```

> 💡 **Keypoints**:
> 1. Declare class-specific constants as `static` members of the class.
> 2. Provide a separate definition in an implementation file if the compiler requires it.
> 3. Only for **static** **constants** of **integral** type, provide an initial value at the point of declaration. Otherwise, provide an initial value of a static member at the point of definition.


### 🎼 Enum Hack

```cpp
class GamePlayer2 {
private:
    enum { NumTurns = 5};  // "the enum hack" - let {NumTurns} be a marker of 5
    int scores[NumTurns];  // valid
}
```

The enum hack is worth knowing about for several reasons. 

1. The enum hack behaves in some ways more like a #define than a const does, and sometimes that's what you want. It's not legal to take the address of an enum, and it's typically not legal to take the address of a #define, either. Also, like #defines, enums never result in unnecessary memory allocation.
2. The enum hack is purely pragmatic. The enum hack is a fundamental technique of template metaprogramming (item 48).

### 🎼 Inline

Use inline functions instead of #defines. 

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

## Term 03: Use `const` Whenever Possible

### 🎼 `const` and Pointers

If the word const appears to the left of the asterisk, what's pointed to is constant; if the word const appears to the right of the asterisk, the pointer itself is constant; if const appears on both sides, both are constant. 

For example:

```cpp
char greeting[] = "Hello";
char* p = greeting;  // non-const pointer, non-const data
const char* p = greeting;  // non-const pointer, const data
char* const p = greeting;  // const pointer, non-const data
const char* const p = greeting;  // const pointer, const data
```

### 🎼 Use `const` to Restrict the User's Behavior 

```cpp
class A
{
public:
    A operator+(const A& a) { return A(); }
};

int main()
{
    A a1, a2;
    a1 + a2 = A();  // This is not expected.
    return 0;
}
```

Where `a1 + a2 = A();` is not expected, because the result of `a1 + a2` is a temporary object, and it is not allowed to assign a value to a temporary object.

To prevent this, you can add `const` to the return value of the `operator+` function:

```cpp
class A
{
public:
    const A operator+(const A& a) { return A(); }
};
```

### 🎼 Const Member Functions

There are two prevailing notions: ***bitwise* constness** (also known as physical constness) and ***logical* constness**.

The bitwise const camp believes that a member function is const if and only if it doesn't modify any of the object's data members (excluding those that are static), i.e., if it doesn't modify any of the bits inside the object.

The nice thing about bitwise constness is that it's easy to detect violations: compilers just look for assignments to data members.

Unfortunately, many member functions that don't act very const pass the bitwise test. For exapmle:

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

> It is worth noting that you should return a `const char&` instead of a `char&` in the `operator[]` function above.

This leads to the notion of **logical constness**. Adherents to this philosophy (and you should be among them) — argue that a const member function might modify some of the bits in the object on which it's invoked, but only in ways that clients cannot detect. For example:

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

### 🎼 Avoiding Duplication in const and Non-const Member Functions

When you have a const and a non-const member function that have essentially identical implementations, you can **avoid code duplication by having the non-const member function call the const member function**. For example:

```cpp
class TextBlock {
public:
    const char& operator[](std::size_t position) const {
        return text[position];
    }
    char& operator[](std::size_t position) {
        return 
            const_cast<char&>(
                static_cast<const TextBlock&>(*this)[position]
            );
    }
}
```

> 💡 **Note**: Do not avoiding duplication by having the const version call the non-const version. A const member function promises never to change the logical state of its object, but a non-const member function makes no such promise.

### 🎼 Things to Remember

- Declaring something `const` helps compilers detect usage errors. `const` can be applied to objects at any scope, to function parameters and return types, and to member functions as a whole.
- Compilers enforce bitwise constness, but you should program using logical constness.
- When `const` and `non-const` member functions have essentially identical implementations, code duplication can be avoided by having the non-const version call the const version.

## Term 04: Make Sure the Objects are initialized before they are used

Always initialize objects before they are used.

### 🎼 Member initialization list

Always use the member initialization list to initialize member objects.

One aspect of C++ that isn't fickle is the order in which an object's data is initialized. This order is always the same: base classes are initialized before derived classes (see also Item 12), and within a class, data members are initialized in the order in which they are declared.

### 🎼 Initialize Static Objects

A **static object** is one that exists from the time it's constructed until the end of the program. Stack and heap-based objects are thus excluded. 

Included are:

 - global objects
 - objects defined at namespace scope
 - objects declared static inside classes
 - objects declared static inside functions
 - objects declared static at file scope 

Static objects inside functions are known as local static objects (because they're local to a function), and the other kinds of static objects are known as non-local static objects. 

Static objects are destroyed when the program exits, i.e., their destructors are called when main finishes executing.

⚠ **Warning**: If initialization of a non-local static object in one translation unit uses a non-local static object in a different translation unit, the object it uses could be uninitialized, because **the relative order of initialization of non-local static objects defined in different translation units is undefined**.

> 💬 Multiple translation units and non-local static objects is generated through implicit template instantiations (which may themselves arise via implicit template instantiations). It's not only impossible to determine the right order of initialization, it's typically not even worth looking for special cases where it is possible to determine the right order.

To avoid the problem of undefined initialization order, you can use **a function-local static object** instead of a non-local static object. These functions return references to the objects they contain. (Aficionados of design patterns will recognize this as a common implementation of the **Singleton Pattern**.)

For example:

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

This approach is founded on C++'s guarantee that **local static objects are initialized when the object's definition is first encountered during a call to that function**. So if you replace direct accesses to non-local static objects with calls to functions that return references to local static objects, you're guaranteed that the references you get back will refer to initialized objects. As a bonus, if you never call a function emulating a non-local static object, you never incur the cost of constructing and destructing the object, something that can't be said for true non-local static objects.

> However, the fact that these functions contain static objects makes them problematic **in multithreaded systems**. Then again, any kind of non-const static object — local or non-local — is trouble waiting to happen in the presence of multiple threads. 
> 
> One way to deal with such trouble is to **manually invoke all the reference-returning functions during the single-threaded startup portion of the program**. This eliminates initialization-related race conditions.

### 🎼 Things to Remember

- Manually initialize objects of built-in type, because C++ only sometimes initializes them itself.
- In a constructor, prefer use of the member initialization list to assignment inside the body of the constructor. List data members in the initialization list in the same order they're declared in the class.
- Avoid initialization order problems across translation units by replacing non-local static objects with local static objects.
