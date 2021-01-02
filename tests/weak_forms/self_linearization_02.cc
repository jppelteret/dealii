// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// Check that (internal) method to perform a tensor product of all
// arguments in a parameter pack work correctly.
//
// This test is adapted from https://stackoverflow.com/a/9145665

#include <deal.II/weak_forms/self_linearizing_forms.h>

#include "../tests.h"


namespace WFT  = WeakForms::SelfLinearization::internal;
namespace WFTP = WeakForms::SelfLinearization::internal::TemplateOuterProduct;


template <typename T>
std::string
print_type()
{
  return WFT::Utilities::TypeHelper<T>::print();
}


template <typename T, typename U>
void
test()
{
  deallog << print_type<T>() << " x " << print_type<U>() << " = "
          << print_type<typename WFTP::OuterProduct<T, U>::type>() << std::endl;
}


struct A
{};
struct B
{};
struct C
{};
struct D
{};
struct E
{};
struct F
{};


int
main()
{
  initlog();

  deallog << "Cartesian product of type lists" << std::endl;
  test<WFTP::TypeList<>, WFTP::TypeList<>>();
  test<WFTP::TypeList<>, WFTP::TypeList<A>>();
  test<WFTP::TypeList<>, WFTP::TypeList<A, B>>();
  test<WFTP::TypeList<A, B>, WFTP::TypeList<>>();
  test<WFTP::TypeList<A>, WFTP::TypeList<B>>();
  test<WFTP::TypeList<A>, WFTP::TypeList<B, C, D>>();
  test<WFTP::TypeList<A, B>, WFTP::TypeList<B, C, D>>();
  test<WFTP::TypeList<A, B, C>, WFTP::TypeList<D>>();
  test<WFTP::TypeList<A, B, C>, WFTP::TypeList<D, E, F>>();

  deallog << "OK" << std::endl;
}
