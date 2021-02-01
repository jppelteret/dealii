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

// Test that the field extractors are configured correctly for the
// (self-linearizing) energy functor
// - Auto-differentiation
// - Scalar fields

#include <deal.II/differentiation/ad.h>

#include <deal.II/weak_forms/weak_forms.h>

#include "../tests.h"

#include "wf_common_tests/energy_functor_utilities.h"


using namespace dealii;


template <int dim, typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int  spacedim = dim;
  constexpr auto ad_typecode =
    Differentiation::AD::NumberTypes::sacado_dfad_dfad;

  // Symbolic types
  const FieldSolution<dim> solution;

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_val  = soln_ss.value();          // Solution value
  const auto soln_grad = soln_ss.gradient();       // Solution gradient
  const auto soln_hess = soln_ss.hessian();        // Solution hessian
  const auto soln_lap  = soln_ss.laplacian();      // Solution laplacian
  const auto soln_d3 = soln_ss.third_derivative(); // Solution third derivative

  // Parameterise energy in terms of all possible operations with the space
  const auto energy = energy_functor(
    "e", "\\Psi", soln_val, soln_grad, soln_hess, soln_lap, soln_d3);
  using ADNumber_t =
    typename decltype(energy)::template ad_type<double, ad_typecode>;

  const auto energy_functor = energy.template value<ADNumber_t, dim, spacedim>(
    [](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
       const std::vector<std::string> &              solution_names,
       const unsigned int                            q_point,
       const ADNumber_t &                            u,
       const Tensor<1, dim, ADNumber_t> &            grad_u,
       const Tensor<2, dim, ADNumber_t> &            hess_u,
       const ADNumber_t &                            lap_u,
       const Tensor<3, dim, ADNumber_t> &d3_u) { return ADNumber_t(0.0); });

  // Look at what we're going to compute
  const SymbolicDecorations decorator;

  deallog << "Energy (ascii):\n"
          << energy_functor.as_ascii(decorator) << std::endl;
  deallog << "Energy (LaTeX):\n"
          << energy_functor.as_latex(decorator) << std::endl;

  print_field_args_and_extractors(energy_functor, decorator);

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  run<2>(subspace_extractor);
  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;

  return 0;
}
