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


// SelfLinearizingEnergyFunctional: Check that (internal) method to
// perform a tensor product of all field solution arguments works correctly.
// - Sub-Space: Vector


#include <deal.II/weak_forms/self_linearizing_forms.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"

#include "wf_common_tests/utilities.h"


namespace WFT = WeakForms::SelfLinearization;


template <typename NumberType, typename... UnaryOpSubSpaceFieldSolution>
void
test(const UnaryOpSubSpaceFieldSolution &... unary_op_subspace_field_soln)
{
  using H =
    WFT::internal::SelfLinearizationHelper<UnaryOpSubSpaceFieldSolution...>;
  using T =
    WFT::SelfLinearizingEnergyFunctional<UnaryOpSubSpaceFieldSolution...>;

  deallog << "Type list: Test function" << std::endl;
  deallog << strip_off_namespace(H::print_type_list_test_function_unary_op())
          << std::endl
          << std::endl;

  deallog << "Type list: Trial solution" << std::endl;
  deallog << strip_off_namespace(H::print_type_list_trial_solution_unary_op())
          << std::endl
          << std::endl;

  deallog << "Linear form pattern (Type list): Test function" << std::endl;
  deallog << strip_off_namespace(T::print_linear_forms_pattern()) << std::endl
          << std::endl;

  deallog
    << "Bilinear form pattern (Outer product type list: Test function X Trial solution"
    << std::endl;
  deallog << strip_off_namespace(T::print_bilinear_forms_pattern()) << std::endl
          << std::endl;

  deallog << "Functor arguments (Type list):" << std::endl;
  deallog << strip_off_namespace(
               T::template print_functor_arguments<NumberType>())
          << std::endl
          << std::endl;

  deallog << "Linear form generator (Type list):" << std::endl;
  deallog << strip_off_namespace(
               T::template print_linear_forms_generator<NumberType>())
          << std::endl
          << std::endl;

  deallog << "Bilinear form generator (Type list):" << std::endl;
  deallog << strip_off_namespace(
               T::template print_bilinear_forms_generator<NumberType>())
          << std::endl
          << std::endl;

  deallog << "OK" << std::endl;
}


template <int dim,
          int spacedim        = dim,
          typename NumberType = double,
          typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  const FieldSolution<dim, spacedim> soln;

  const auto soln_ss                    = soln[subspace_extractor];
  const auto value_soln_ss              = soln_ss.value();
  const auto gradient_soln_ss           = soln_ss.gradient();
  const auto symmetric_gradient_soln_ss = soln_ss.symmetric_gradient();
  const auto divergence_soln_ss         = soln_ss.divergence();
  const auto curl_soln_ss               = soln_ss.curl();
  const auto hessian_soln_ss            = soln_ss.hessian();
  const auto third_derivative_soln_ss   = soln_ss.third_derivative();

  // We can compose functions with an arbitrary number of input
  // arguments, all stemming from the same solution space but
  // using different differential operators.
  test<NumberType>(value_soln_ss);
  test<NumberType>(value_soln_ss, gradient_soln_ss);
  test<NumberType>(value_soln_ss, gradient_soln_ss, symmetric_gradient_soln_ss);
  test<NumberType>(value_soln_ss,
                   gradient_soln_ss,
                   symmetric_gradient_soln_ss,
                   divergence_soln_ss);
  test<NumberType>(value_soln_ss,
                   gradient_soln_ss,
                   symmetric_gradient_soln_ss,
                   divergence_soln_ss,
                   curl_soln_ss);
  test<NumberType>(value_soln_ss,
                   gradient_soln_ss,
                   symmetric_gradient_soln_ss,
                   divergence_soln_ss,
                   curl_soln_ss,
                   hessian_soln_ss);
  test<NumberType>(value_soln_ss,
                   gradient_soln_ss,
                   symmetric_gradient_soln_ss,
                   divergence_soln_ss,
                   curl_soln_ss,
                   hessian_soln_ss,
                   third_derivative_soln_ss);

  // This should not compile, because it implies that
  // we can use the same argument twice, which we do
  // not want.
  //
  // test(value_soln_ss, value_soln_ss);

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  const WeakForms::SubSpaceExtractors::Vector subspace_extractor(0, "V", "V");

  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;
}
