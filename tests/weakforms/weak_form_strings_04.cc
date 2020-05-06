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


// Check weak form stringization and printing
// - Linear, bilinear forms


#include <deal.II/weakforms/bilinear_forms.h>
#include <deal.II/weakforms/linear_forms.h>
#include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/symbolic_info.h>

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

  const TestFunction<dim, spacedim>  test(naming_ascii, naming_latex);
  const TrialSolution<dim, spacedim> trial(naming_ascii, naming_latex);
  const FieldSolution<dim, spacedim> soln(naming_ascii, naming_latex);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Linear form: " << linear_form(test, soln).as_ascii()
            << std::endl; // Note: Not really permissible

    deallog << "Bilinear form: " << bilinear_form(test, soln, trial).as_ascii()
            << std::endl; // Note: Not really permissible

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Linear form: " << linear_form(test, soln).as_latex()
            << std::endl; // Note: Not really permissible

    deallog << "Bilinear form: " << bilinear_form(test, soln, trial).as_latex()
            << std::endl; // Note: Not really permissible

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
