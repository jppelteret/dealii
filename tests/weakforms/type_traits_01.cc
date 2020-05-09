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


// Check type traits for spaces


#include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/type_traits.h>

#include "../tests.h"


int
main()
{
  initlog();

  using namespace WeakForms;

  constexpr int dim = 2;
  constexpr int spacedim = 2;

  using test_t = TestFunction<dim, spacedim>;
  using trial_t = TrialSolution<dim, spacedim>;
  using soln_t = FieldSolution<dim, spacedim>;

  deallog << std::boolalpha;

  deallog << "is_test_function()"<< std::endl;
  deallog << is_test_function<test_t>::value << std::endl;
  deallog << is_test_function<trial_t>::value << std::endl;
  deallog << is_test_function<soln_t>::value << std::endl;

  deallog << std::endl;
  
  deallog << "is_trial_solution()"<< std::endl;
  deallog << is_trial_solution<test_t>::value << std::endl;
  deallog << is_trial_solution<trial_t>::value << std::endl;
  deallog << is_trial_solution<soln_t>::value << std::endl;

  deallog << std::endl;
  
  deallog << "is_field_solution()"<< std::endl;
  deallog << is_field_solution<test_t>::value << std::endl;
  deallog << is_field_solution<trial_t>::value << std::endl;
  deallog << is_field_solution<soln_t>::value << std::endl;

  deallog << std::endl;

  deallog << "OK" << std::endl;
}
