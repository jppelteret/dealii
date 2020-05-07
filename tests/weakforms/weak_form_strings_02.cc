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


// Check functor form stringization and printing
// - Functors


#include <deal.II/weakforms/functors.h>
// #include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/symbolic_info.h>
#include <deal.II/weakforms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicNamesAscii naming_ascii;
  const SymbolicNamesLaTeX naming_latex;

  // TestFunction<dim, spacedim>  test(naming_ascii, naming_latex);
  // TrialSolution<dim, spacedim> trial(naming_ascii, naming_latex);
  // FieldSolution<dim, spacedim> soln(naming_ascii, naming_latex);

  const ScalarFunctor<NumberType> scalar ("U", "U", naming_ascii, naming_latex);
  const TensorFunctor<2,dim,NumberType> tensor ("T", "T", naming_ascii, naming_latex);

  const auto s = value(scalar,[](const unsigned int){return 1.0;});
  const auto T = value(tensor,[](const unsigned int){return Tensor<2,dim,NumberType>();});

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "FUNCTOR CREATION" << std::endl;
    deallog << "Scalar: " << scalar.as_ascii() << std::endl;
    deallog << "Tensor: " << tensor.as_ascii() << std::endl;
  //   deallog << "Function: " << soln.as_ascii() << std::endl;

    deallog << "FUNCTOR VALUE SETTING" << std::endl;
    deallog << "Scalar: " << s.as_ascii() << " ; val: " << s(0) << std::endl;
    deallog << "Tensor: " << T.as_ascii() << " ; val: " << T(0) << std::endl;
  //   deallog << "Tensor: " << trial.as_ascii() << std::endl;
  //   deallog << "Function: " << soln.as_ascii() << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "FUNCTOR CREATION" << std::endl;
    deallog << "Scalar: " << scalar.as_latex() << std::endl;
    deallog << "Tensor: " << tensor.as_latex() << std::endl;
  //   deallog << "Function: " << soln.as_latex() << std::endl;

    deallog << "FUNCTOR VALUE SETTING" << std::endl;
    deallog << "Scalar: " << s.as_latex() << " ; val: " << s(0) << std::endl;
    deallog << "Tensor: " << T.as_latex() << " ; val: " << T(0) << std::endl;
  //   deallog << "Function: " << soln.as_latex() << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
