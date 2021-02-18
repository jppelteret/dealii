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
// - Auto-differentiable energy functor
// - Sub-Space: Tensor

#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/weak_forms/energy_functor.h>
#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/symbolic_operators.h>

#include "../tests.h"


template <int dim,
          int spacedim        = dim,
          typename NumberType = double,
          typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using namespace Differentiation;

  // Customise the naming conventions, if we wish to.
  const SymbolicDecorations decorator;

  const FieldSolution<dim> solution;

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_val  = soln_ss.value();      // Solution value
  const auto soln_grad = soln_ss.gradient();   // Solution gradient
  const auto soln_div  = soln_ss.divergence(); // Solution divergence

  const auto energy_1 = energy_functor("e", "\\Psi", soln_val);
  const auto energy_2 = energy_functor("e", "\\Psi", soln_val, soln_grad);
  const auto energy_3 =
    energy_functor("e", "\\Psi", soln_val, soln_grad, soln_div);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "(v): " << energy_1.as_ascii(decorator) << std::endl;
    deallog << "(v,g): " << energy_2.as_ascii(decorator) << std::endl;
    deallog << "(v,g,d): " << energy_3.as_ascii(decorator) << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "(v): " << energy_1.as_latex(decorator) << std::endl;
    deallog << "(v,g): " << energy_2.as_latex(decorator) << std::endl;
    deallog << "(v,g,d): " << energy_3.as_latex(decorator) << std::endl;

    deallog << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  constexpr int                                     rank = 2;
  const WeakForms::SubSpaceExtractors::Tensor<rank> subspace_extractor(
    0, "T", "\\mathbf{T}");

  run<3>(subspace_extractor);

  deallog << "OK" << std::endl;
}
