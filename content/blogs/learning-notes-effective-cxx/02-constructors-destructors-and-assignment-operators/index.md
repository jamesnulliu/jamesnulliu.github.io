---
title: "02 | Constructors, Destructorsm and Assignment Operators"
date: 2024-06-29T00:02:00+08:00
lastmod: 2024-06-29T18:00:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
categories:
    - notes
tags:
    - c++
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

## Term 05 Know what functions C++ silently writes and calls

A compiler will automatically generate four functions for a class if the class does not define them by itself:

- default constructor
- copy constructor
- copy assignment operator
- destructor

The destructor compiler automatically generates is non-virtual, unless that the base class's destructor is virtual. (The virtualness mainly comes from the base class.)

If a class has reference members or const members, you have to define a copy assignment operator by yourself. The compiler would not generate a default copy assignment operator.

If some base class declares that its copy assignment operator is private, the compiler will refuse to generate a default copy assignment operator for its derived class.

## Term 06 Explicitly disallow the use of compiler-generated functions you do not want

One way to disallow the use of copy constructor and copy assignment (more specifically, the compiler-generated ones) is to inherit a class that declared private copy constructor and copy assignment operator.

## Term 07 Declare destructor virtual in polymorphic base classes

When deleting a base pointer pointing to a instance of a derived class, if destructor is declared non-virtual in base class, only the memory of base part would be released.

Consider about a class `Point`:

```cpp
class Point {
public:
	Point(int xCoord, int yCoord);
	~Point();
private:
	int x, y;
}
```

If "int" takes 32 bits, a `Point` instance takes 64 bits, and can be easily passed to other languages such as C and FORTRAN.

However, if we declare `~Point()` as a virtual function, a **vptr** (virtual table pointer) would be added to the instance. That causes the instance takes upto 128 bits (2 32-bit integer and 1 64-bit pointer). Moreover, since C and FORTRAN does not have vptr, the class is not portable any more.

To create an abstract class, you could (I think you'd better declare other functions pure virtual) declare a pure virtual destructor; But you has to offer a definition for the destructor outside the class (maybe in a cpp file).  
The reason is that when an instance of a derived class is dectructed, the destructor of the most derived class is called, and each base class of them is called after. So there is an actor that the pure virtual destructor (of base class) is called inside the derived class.

### Sum up

- Do not delcare a function virtual with no reason.
- Do not inherit frome a class that does not have a virtual destructor (e.g., std::string, std::set, std::vector, std::map) when the class is not designed for polymorphism.

## Term 08 Prevent exceptions from leaving destructors

Trow exceptions out from a destructor is not encouraged by C++.  

Suppose there are several objects in a block. At the end of the block, all the objects are destroyed automatically.  
If destructor of the first object throws an exception, every thing is okay; Destructors of the other objects would be called porperly.  
However, if the second destructor also throws an exception, the program would either be terminated or cause an undefined behavior (that would be fatal). 

A good strategy is to give the chance to users that they can handle the exceptions themselves.

In the following example code, db is an instance of class `DBConnect`, and meanwhile it is a member of class `DBConn`.  
Before calling `~DBConn()` automatically, user (instead of the compiler) should call `DBConn::close()` at first and handle the possible exception thrown by `DBConnect::close()`.

```cpp
class DBConn {
public:
	...
	void close() // A close function for users to use
	{
		db.close();
		closed = true;
	}
	
	~DBConn()
	{
		if(!closed){
			try{
				db.close();
			} catch(...) {
				Log the faliure;
			}
		}
	}
	
private:
	DBConnection db;
	bool closed;
}
```

### Sum up
- Do not give any chance for an exception to leave a destructor. Destructor should catch and handle the exceptions inside itself.
- Offer a function to let user handle the exception that inside the destructor.

## Term 09: Never call virtual functions during construction or distruction.
In C++, when constructing a derived class, the base part is constructed first; And during the construction of base part, the vptr is still pointing at the base class. This means, if you invoke a virtual function in the constructor of a base class, when you create an instance of a derived class, the actual called virtual function is the base version, not the overrided one.

## Term 10: Have assignment operator return a reference to `*this`

```cpp
class Widget
{
public:
	Widget& operator+=(const Widget& rhs)
	{
		...;
		return *this;
	}
	Widget& operator=(int rhs)
	{
		...;
		return *this;
	}
private:
	Bitmap* pb;
};
```

## Term 11: Handle assignment to self in `operator=`
Self-assignment could cause a question that the resources are released before they are assigned.

Traditionally, **identity test** can check whether there is an assignment to self:

```cpp
Widget& Widget::operator=(const Widget& rhs)
{
	if(this == &rhs) return *this;  // Identity test
	delete pb;
	pb = new Bitmap(*rhs.pb);
	return *this;
}
```

However, if the exception occurs (either because the memory is not enough when allocation or the copy constructor throws one exception), the pointer `pb` would ultimately points to a deleted Bitmap, and that is harmful.

Nowadays more people tends to care for **exception safety** rather than **self-allocation safety**. For example:

```cpp
Widget& Widget::operator=(const Widget& rhs)
{
	Bitmap* pOrigin = pb;
	pb = new Bitmap(*rhs.pb);
	delete pOrigin;
	return *this;
}
```

Even if without identity test, self-assignment can be handled, and `pb` has no chance to point to a deleted Bitmap.

Identity test can be put back to the begin of the funtion; But that may lower the efficiency, since self-assignment does not happen so much.

There is another way called **copy and swap** technic. For example:

```cpp
Widget& Widget::operator=(const Widget& rhs)
{
	Widget temp(rhs);
	this->swap(temp);
	return *this;
}
```

Or:

```cpp
Widget& Widget::operator=(Widget rhs)
{
	this->swap(thd);
	return *this;
}
```

The second way sacrificces clearity; However, because it moves "copy" action from the function body to "parameter-constructing stage", sometimes the compiler could generate more efficient codes.

## Term 12: Copy all parts of an object.
Compiler would not warn you if there is a particial copy, and do not let that happen.

Copy constructor of a derived class should invoke the copy constructor of base class:

```cpp
class Base
{
public:
	Base(const Base& rhs)
		: name(rhs.name),
			lastTransaction(rhs.lastTransaction)
	{
		log();
	}
private:
	std::string name;
	Date lastTransaction;
};

class Derived : Base
{
public:
	Derived(const Derived& rhs)
		: Base(rhs),  // Invoke the copy constructor of base class
			priority(rhs.priority)
	{
		log();
	}
private:
	int priority;
}
```

Do not have copy assignment operator call copy constructor, vice versa.  
If you want, you can write a function `init()` additionally and call it in both functions.