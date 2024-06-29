---
title: "03 | Resource Management"
date: 2024-06-29T00:03:00+08:00
lastmod: 2024-06-29T18:16:00+08:00
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

## Term 13: Use objects to manage resources

Do not make multiple `std::auto_ptr`s point to the same object.  
If one is destroyed, the object would be released automatically.

Simply put, do not use `std::auto_ptr`.

Use **RCSP (reference-counting smart pointer)** to manage resources:  
`std::shared_ptr`

Remember: both `auto_ptr` and `shared_ptr` use `delete` instead of `delete[]`. As a result, do not write `std::shared_ptr<int> spi(new int[1024])`.

## Term 14: Think carefully about copying behavior in resource-managing classes.
Term 13 introduces such concept: **Resource Acquisition Is Initialization, RAII**. This concept is performed when `std::auto_ptr` and `std::shared_ptr` manage heap-based resource.  
Not all resource is  heap-based, and it is usual that RCSP is not suitable resource handlers for these resource.  
That is why sometimes you should write your own resource-managing classes.

Suppose we use a mutex object with class `Mutex`, and there are two functions `lock()` and `unlock()`. In order not to forget to unlock any locked mutex, it is feasible to write a class `Lock` to manage the mutex. Example is shown as following:

```cpp
class Lock
{
public:
	explicit Lock(Mutex* pm) : mutexPtr(pm)
	{
		lock(mutexPtr);
	}
	~Lock()
	{
		unlock(mutexPtr);
	}
private:
	Mutex* mutexPtr;
}
```

If we copy a `Lock` object to another, in most cases you would choose one of the following four options.

- **To Forbid Copying**. By following term 6, you can make class `Lock` inherit from a base class whose copy constructor is declared private.
- **Use Reference-Count for Underlying Resource**. Use `std::shared_ptr<Mutex>` to manage `Lock::mutexPtr` in stead of `Mutex*`. However, one question is that what a `std::shared_ptr` do is when reference-count equals to 0, the underlying pointer is deleted, and that is not what we want. We want to call function `unlock`. The lucky thing is that `std::share_ptr` allow users to specify a **deleter**, so the class `Lock` can be written as follow:  
  ```cpp
  class Lock
  {
	public:
	  explicit Lock(Mutex* pm) : mutexPtr(pm, unlock) // Use func {unlock} to sepecify a deleter and initialize a std::shared_ptr
	  {
		  lock(mutexPtr.get());
	  }
	  // Destructor is omitted because {mutexPtr} would automatically invoke func {unlock}.
	private:
	  std::shared_ptr<Mutex> mutexPtr;
  }
  ```
- **Deep Copying**.
- **Transfer Ownership**. For example: `std::auto_ptr`.

## Term 15: Provide access to raw resources in resource-managing classes
`std::shared_ptr` and `std::auto_ptr` both provide `get()` methods to give access to the underlying raw pointers.

```cpp
class Font
{
public:
	explicit Font(FontHandle fh) : f(fh) {}
	~Font() { releaseFont(f); }
	FontHandle get() const { return f; }  // Explicit conversion
	operator FontHandle() const { return f; }  // Implicit conversion
private:
	FontHandle f;  // Raw font resource
}
```

Implicit conversion could be dangerous.

## Term 16: Use the same form in corresponding uses of new and delete.

## Term 17: Store new objects in smart pointers in standalone statements.
Suppose there is a function:

```cpp
void processWidget(std::shared_ptr<Widget> pw, int priority);
```

Following call of the function is invalid:

```cpp
void processWidget(new Widget, priority());
```

The reason is that construction of `std::shared_ptr` is declared explicitly. If we adjust the statement to:

```cpp
void processWidget(std::share_ptr<Widget>(new Widget), priority());
```

This could cause memory leak. The above statement has to do 3 things:

- Call `priority()`
- Run `new Widget`
- Call constructor of `std::shared_ptr`

In C++, in order to generate more efficient code, the execution sequence could be: 

1. Run `new Widget`
2. Call `priority()`
3. Call constructor of `std::share_ptr`

In this situation, if an exception is thrown when `priority()` is called, the resource accuired by `new Widget` would not be free properly. This is memory leak.

To avoid this, write as following:

```cpp
std::shared_ptr<Widget> pw(new Widget);
processWidget(pw, priority());
```