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
// - Spaces


#include <deal.II/weakforms/binary_operators.h>
#include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/symbolic_info.h>
#include <deal.II/weakforms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicNamesAscii naming_ascii;
  const SymbolicNamesLaTeX naming_latex;

  const std::string            field_string = "phi";
  const std::string            field_latex  = "\\varphi";
  TestFunction<dim, spacedim>  test(field_string,
                                   field_latex,
                                   naming_ascii,
                                   naming_latex);
  TrialSolution<dim, spacedim> trial(field_string,
                                     field_latex,
                                     naming_ascii,
                                     naming_latex);
  FieldSolution<dim, spacedim> soln(field_string,
                                    field_latex,
                                    naming_ascii,
                                    naming_latex);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test.as_ascii() << std::endl;
    deallog << "Trial solution: " << trial.as_ascii() << std::endl;
    deallog << "Solution: " << soln.as_ascii() << std::endl << std::endl;

    deallog << "OPERATIONS WITH SPACES" << std::endl;
    deallog << "Addition: " << (trial + soln).as_ascii() << std::endl
            << std::endl; // Note: Not really permissible
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test.as_latex() << std::endl;
    deallog << "Trial solution: " << trial.as_latex() << std::endl;
    deallog << "Solution: " << soln.as_latex() << std::endl << std::endl;

    deallog << "OPERATIONS WITH SPACES" << std::endl;
    deallog << "Addition: " << (trial + soln).as_latex() << std::endl
            << std::endl; // Note: Not really permissible
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
