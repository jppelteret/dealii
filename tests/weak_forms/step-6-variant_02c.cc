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

// Laplace problem: Assembly using weak forms
// This test replicates step-6 exactly.

#include <deal.II/weak_forms/weak_forms.h>

#include "../tests.h"

#include "wf_common_tests/step-6.h"


using namespace dealii;


template <int dim>
class Step6 : public Step6_Base<dim>
{
public:
  Step6();

protected:
  void
  assemble_system() override;
};


template <int dim>
Step6<dim>::Step6()
  : Step6_Base<dim>()
{}


template <int dim>
void
Step6<dim>::assemble_system()
{
  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim>  test(decorator);
  const TrialSolution<dim> trial(decorator);
  const ScalarFunctor      mat_coeff("c", "c", decorator);
  const ScalarFunctor      rhs_coeff("s", "s", decorator);

  const auto test_val       = value(test);     // Shape function value
  const auto test_grad      = gradient(test);  // Shape function gradient
  const auto trial_grad     = gradient(trial); // Shape function gradient
  const auto mat_coeff_func = value<double>(mat_coeff, [](const unsigned int) {
    return 1.0;
  }); // Coefficient
  const auto rhs_coeff_func = value<double>(rhs_coeff, [](const unsigned int) {
    return 1.0;
  }); // Coefficient

  MatrixBasedAssembler<dim> assembler;
  assembler += bilinear_form(test_grad, mat_coeff_func, trial_grad).dV();
  assembler -= linear_form(test_val, rhs_coeff_func).dV();

  // Look at what we're going to compute
  std::cout << "Weak form (ascii):\n" << assembler.as_ascii() << std::endl;
  std::cout << "Weak form (LaTeX):\n" << assembler.as_latex() << std::endl;

  // Now we pass in concrete objects to get data from
  // and assemble into.
  assembler.assemble(this->system_matrix,
                     this->system_rhs,
                     this->constraints,
                     this->dof_handler,
                     this->qf_cell);
}


int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  try
    {
      Step6<2> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  deallog << "OK" << std::endl;

  return 0;
}
