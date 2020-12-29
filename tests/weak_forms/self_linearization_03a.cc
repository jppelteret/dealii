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
// field solution arguments works correctly.
// - Sub-Space: Scalar


#include <deal.II/weak_forms/self_linearizing_forms.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/unary_operators.h>

#include <regex>

#include "../tests.h"


namespace WFT = WeakForms::SelfLinearization::internal;


std::string
strip_off_namespace(std::string demangled_type)
{
  const std::vector<std::string> names{
    "dealii::WeakForms::Operators::", "dealii::WeakForms::", "dealii::"};

  for (const auto &name : names)
    demangled_type = std::regex_replace(demangled_type, std::regex(name), "");

  return demangled_type;
}


template <typename NumberType, typename... UnaryOpSubSpaceFieldSolution>
void
test(const UnaryOpSubSpaceFieldSolution &... unary_op_subspace_field_soln)
{
  using T = WFT::SelfLinearizationHelper<UnaryOpSubSpaceFieldSolution...>;

  deallog << "Type list: Field solution" << std::endl;
  deallog << strip_off_namespace(T::print_type_list_field_solution_unary_op())
          << std::endl
          << std::endl;

  deallog << "Type list: Functor input arguments" << std::endl;
  deallog << strip_off_namespace(
               T::template print_type_list_value_type<NumberType>())
          << std::endl
          << std::endl;

  deallog << "Outer product type list: Field solution X Field solution"
          << std::endl;
  deallog << strip_off_namespace(
               T::print_field_solution_unary_op_outer_product_type())
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

  const auto soln_ss                  = soln[subspace_extractor];
  const auto value_soln_ss            = soln_ss.value();
  const auto gradient_soln_ss         = soln_ss.gradient();
  const auto hessian_soln_ss          = soln_ss.hessian();
  const auto laplacian_soln_ss        = soln_ss.laplacian();
  const auto third_derivative_soln_ss = soln_ss.third_derivative();

  // We can compose functions with an arbitrary number of input
  // arguments, all stemming from the same solution space but
  // using different differential operators.
  test<NumberType>(value_soln_ss);
  test<NumberType>(value_soln_ss, gradient_soln_ss);
  test<NumberType>(value_soln_ss, gradient_soln_ss, hessian_soln_ss);
  test<NumberType>(value_soln_ss,
                   gradient_soln_ss,
                   hessian_soln_ss,
                   laplacian_soln_ss);
  test<NumberType>(value_soln_ss,
                   gradient_soln_ss,
                   hessian_soln_ss,
                   laplacian_soln_ss,
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

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;
}
