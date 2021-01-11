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
// - Sub-Space: SymmetricTensor


#include <deal.II/weak_forms/self_linearizing_forms.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"

#include "wf_common_tests/utilities.h"


namespace WFT = WeakForms::SelfLinearization::internal;


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
               T::template print_type_list_functor_arguments<NumberType>())
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

  const auto soln_ss            = soln[subspace_extractor];
  const auto value_soln_ss      = soln_ss.value();
  const auto divergence_soln_ss = soln_ss.divergence();

  // We can compose functions with an arbitrary number of input
  // arguments, all stemming from the same solution space but
  // using different differential operators.
  test<NumberType>(value_soln_ss);
  test<NumberType>(value_soln_ss, divergence_soln_ss);

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

  const WeakForms::SubSpaceExtractors::SymmetricTensor<2> subspace_extractor(
    0, "S", "S");

  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;
}