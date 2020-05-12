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


// Check type traits for space function operations


#include <deal.II/weak_forms/binary_operators.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"


int
main()
{
  initlog();

  using namespace WeakForms;

  constexpr int dim      = 2;
  constexpr int spacedim = 2;

  using test_t  = TestFunction<dim, spacedim>;
  using trial_t = TrialSolution<dim, spacedim>;
  using soln_t  = FieldSolution<dim, spacedim>;

  using test_val_t  = decltype(value(std::declval<test_t>()));
  using trial_val_t = decltype(value(std::declval<trial_t>()));
  using soln_val_t  = decltype(value(std::declval<soln_t>()));

  using test_grad_t  = decltype(gradient(std::declval<test_t>()));
  using trial_grad_t = decltype(gradient(std::declval<trial_t>()));
  using soln_grad_t  = decltype(gradient(std::declval<soln_t>()));

  const SymbolicDecorations decorator;

  const TestFunction<dim, spacedim>  test(decorator);
  const TrialSolution<dim, spacedim> trial(decorator);
  const FieldSolution<dim, spacedim> soln(decorator);

  const auto test_val  = value(test);
  const auto trial_val = value(trial);
  const auto soln_val  = value(soln);
  const auto soln_grad = gradient(soln);

  deallog << std::boolalpha;

  // Values: <something> * <solution value>
  {
    LogStream::Prefix prefix("Value: RHS");

    deallog << "is_test_function()" << std::endl;
    deallog << is_test_function<decltype(std::declval<test_val_t>() *
                                         std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_test_function<decltype(std::declval<trial_val_t>() *
                                         std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_test_function<decltype(std::declval<soln_val_t>() *
                                         std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution()" << std::endl;
    deallog << is_trial_solution<decltype(std::declval<test_val_t>() *
                                          std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_trial_solution<decltype(std::declval<trial_val_t>() *
                                          std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_trial_solution<decltype(std::declval<soln_val_t>() *
                                          std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution()" << std::endl;
    deallog << is_field_solution<decltype(std::declval<test_val_t>() *
                                          std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_field_solution<decltype(std::declval<trial_val_t>() *
                                          std::declval<soln_val_t>())>::value
            << std::endl;
    deallog << is_field_solution<decltype(std::declval<soln_val_t>() *
                                          std::declval<soln_val_t>())>::value
            << std::endl;

    deallog << std::endl;
  }

  // Values: <solution value> * <something>
  {
    LogStream::Prefix prefix("Value: LHS");

    deallog << "is_test_function()" << std::endl;
    deallog << is_test_function<decltype(soln_val * test_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_val * trial_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_val * soln_val)>::value
            << std::endl;

    // For some reason this doesn't work:
    // deallog <<
    // is_test_function<decltype(std::declval<soln_val_t>*std::declval<test_val_t>())>::value
    // << std::endl; deallog <<
    // is_test_function<decltype(std::declval<soln_val_t>*std::declval<trial_val_t>())>::value
    // << std::endl; deallog <<
    // is_test_function<decltype(std::declval<soln_val_t>*std::declval<soln_val_t>())>::value
    // << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution()" << std::endl;
    deallog << is_trial_solution<decltype(soln_val * test_val)>::value
            << std::endl;
    deallog << is_trial_solution<decltype(soln_val * trial_val)>::value
            << std::endl;
    deallog << is_trial_solution<decltype(soln_val * soln_val)>::value
            << std::endl;

    // For some reason this doesn't work:
    // deallog <<
    // is_trial_solution<decltype(std::declval<soln_val_t>*std::declval<test_val_t>())>::value
    // << std::endl; deallog <<
    // is_trial_solution<decltype(std::declval<soln_val_t>*std::declval<trial_val_t>())>::value
    // << std::endl; deallog <<
    // is_trial_solution<decltype(std::declval<soln_val_t>*std::declval<soln_val_t>())>::value
    // << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution()" << std::endl;
    deallog << is_field_solution<decltype(soln_val * test_val)>::value
            << std::endl;
    deallog << is_field_solution<decltype(soln_val * trial_val)>::value
            << std::endl;
    deallog << is_field_solution<decltype(soln_val * soln_val)>::value
            << std::endl;

    // For some reason this doesn't work:
    // deallog <<
    // is_field_solution<decltype(std::declval<soln_val_t>*std::declval<test_val_t>())>::value
    // << std::endl; deallog <<
    // is_field_solution<decltype(std::declval<soln_val_t>*std::declval<trial_val_t>())>::value
    // << std::endl; deallog <<
    // is_field_solution<decltype(std::declval<soln_val_t>*std::declval<soln_val_t>())>::value
    // << std::endl;

    deallog << std::endl;
  }

  // Gradients <something> * <solution gradient>
  {
    LogStream::Prefix prefix("Gradient: RHS");

    deallog << "is_test_function()" << std::endl;
    deallog << is_test_function<decltype(std::declval<test_val_t>() *
                                         std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_test_function<decltype(std::declval<trial_val_t>() *
                                         std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_test_function<decltype(std::declval<soln_val_t>() *
                                         std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_test_function<decltype(std::declval<soln_grad_t>() *
                                         std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution()" << std::endl;
    deallog << is_trial_solution<decltype(std::declval<test_val_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_trial_solution<decltype(std::declval<trial_val_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_trial_solution<decltype(std::declval<soln_val_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_trial_solution<decltype(std::declval<soln_grad_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution()" << std::endl;
    deallog << is_field_solution<decltype(std::declval<test_val_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_field_solution<decltype(std::declval<trial_val_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_field_solution<decltype(std::declval<soln_val_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;
    deallog << is_field_solution<decltype(std::declval<soln_grad_t>() *
                                          std::declval<soln_grad_t>())>::value
            << std::endl;

    deallog << std::endl;
  }

  // Gradients: <solution gradient> * <something>
  {
    LogStream::Prefix prefix("Value: LHS");

    deallog << "is_test_function()" << std::endl;
    deallog << is_test_function<decltype(soln_grad * test_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * trial_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * soln_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * soln_grad)>::value
            << std::endl;

    // For some reason this doesn't work:
    // deallog << is_test_function<decltype(std::declval<soln_grad_t> *
    //                                      std::declval<test_val_t>())>::value
    //         << std::endl;
    // deallog << is_test_function<decltype(std::declval<soln_grad_t> *
    //                                      std::declval<trial_val_t>())>::value
    //         << std::endl;
    // deallog << is_test_function<decltype(std::declval<soln_grad_t> *
    //                                      std::declval<soln_val_t>())>::value
    //         << std::endl;

    deallog << std::endl;

    deallog << "is_trial_solution()" << std::endl;
    deallog << is_test_function<decltype(soln_grad * test_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * trial_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * soln_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * soln_grad)>::value
            << std::endl;

    // For some reason this doesn't work:
    // deallog << is_trial_solution<decltype(std::declval<soln_grad_t> *
    //                                       std::declval<test_val_t>())>::value
    //         << std::endl;
    // deallog << is_trial_solution<decltype(std::declval<soln_grad_t> *
    //                                       std::declval<trial_val_t>())>::value
    //         << std::endl;
    // deallog << is_trial_solution<decltype(std::declval<soln_grad_t> *
    //                                       std::declval<soln_val_t>())>::value
    //         << std::endl;

    deallog << std::endl;

    deallog << "is_field_solution()" << std::endl;
    deallog << is_test_function<decltype(soln_grad * test_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * trial_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * soln_val)>::value
            << std::endl;
    deallog << is_test_function<decltype(soln_grad * soln_grad)>::value
            << std::endl;

    // For some reason this doesn't work:
    // deallog << is_field_solution<decltype(std::declval<soln_grad_t> *
    //                                       std::declval<test_val_t>())>::value
    //         << std::endl;
    // deallog << is_field_solution<decltype(std::declval<soln_grad_t> *
    //                                       std::declval<trial_val_t>())>::value
    //         << std::endl;
    // deallog << is_field_solution<decltype(std::declval<soln_grad_t> *
    //                                       std::declval<soln_val_t>())>::value
    //         << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl;
}
