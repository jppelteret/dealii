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


// Check assembly
// - Functor helpers: Symmetric Tensor


#include <deal.II/weak_forms/assembler.h>
#include <deal.II/weak_forms/bilinear_forms.h>
#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/linear_forms.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/symbolic_decorations.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  auto print_assembler = [](const MatrixBasedAssembler<dim> &assembler) {
    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    deallog << "Weak form (ascii):\n"
            << assembler.as_ascii(decorator) << std::endl;
    deallog << "Weak form (LaTeX):\n"
            << assembler.as_latex(decorator) << std::endl;
    deallog << "OK" << std::endl;
  };


  // Rank-2 symmetric tensor functor
  {
    LogStream::Prefix prefix("SymmTensor2");

    const auto S2 =
      constant_symmetric_tensor<2, dim>(unit_symmetric_tensor<dim>());

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim>  test;
    const TrialSolution<dim> trial;

    const WeakForms::SubSpaceExtractors::Vector subspace_extractor(
      0, "u", "\\mathbf{u}");

    const auto test_ss   = test[subspace_extractor];
    const auto test_grad = test_ss.gradient();

    const auto trial_ss  = trial[subspace_extractor];
    const auto trial_div = trial_ss.divergence();

    MatrixBasedAssembler<dim> assembler;
    assembler += bilinear_form(test_grad, S2, trial_div).dV() +
                 linear_form(test_grad, S2).dA();
    print_assembler(assembler);
  }

  // Rank-4 symmetric tensor functor
  {
    LogStream::Prefix prefix("SymmTensor4");

    const auto S4 = constant_symmetric_tensor<4, dim>(identity_tensor<dim>());

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim>  test;
    const TrialSolution<dim> trial;

    const WeakForms::SubSpaceExtractors::SymmetricTensor<2> subspace_extractor(
      0, "S", "\\mathbf{S}");

    const auto test_ss  = test[subspace_extractor];
    const auto test_val = test_ss.value();

    const auto trial_ss  = trial[subspace_extractor];
    const auto trial_val = trial_ss.value();

    MatrixBasedAssembler<dim> assembler;
    assembler += bilinear_form(test_val, S4, trial_val).dV();
    print_assembler(assembler);
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
