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

// Laplace problem: Assembly using self-linearizing weak form in conjunction
// with symbolic differentiation.
// This test replicates step-6 exactly.
// - Optimizer type: LLVM
// - Optimization method: All

#include <deal.II/differentiation/sd.h>

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
  using namespace Differentiation;

  constexpr int spacedim = dim;
  using SDNumber_t       = typename Differentiation::SD::Expression;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim>  test;
  const FieldSolution<dim> solution;
  const ScalarFunctor      rhs_coeff("s", "s");

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  const auto test_ss  = test[subspace_extractor];
  const auto test_val = test_ss.value();

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_grad = soln_ss.gradient(); // Solution gradient

  const auto energy_func = energy_functor("e", "\\Psi", soln_grad);

  const SDNumber_t coefficient("c");
  const auto energy = energy_func.template value<SDNumber_t, dim, spacedim>(
    [coefficient](const Tensor<1, spacedim, SDNumber_t> &grad_u) {
      return 0.5 * coefficient * scalar_product(grad_u, grad_u);
    },
    [coefficient](const Tensor<1, spacedim, SDNumber_t> &grad_u) {
      return Differentiation::SD::make_symbol_map(coefficient);
    },
    [coefficient](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                  const std::vector<std::string> &              solution_names,
                  const unsigned int                            q_point) {
      const Point<spacedim> &p = scratch_data.get_quadrature_points()[q_point];
      const double           c = (p.square() < 0.5 * 0.5 ? 20.0 : 1.0);
      return Differentiation::SD::make_substitution_map(coefficient, c);
    },
    Differentiation::SD::OptimizerType::llvm,
    Differentiation::SD::OptimizationFlags::optimize_all,
    UpdateFlags::update_quadrature_points);

  const auto rhs_coeff_func = rhs_coeff.template value<double, dim, spacedim>(
    [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
      return 1.0;
    });

  MatrixBasedAssembler<dim> assembler;
  assembler += energy_functional_form(energy, soln_grad).dV();
  assembler -= linear_form(test_val, rhs_coeff_func).dV(); // RHS contribution

  // Look at what we're going to compute
  const SymbolicDecorations decorator;
  static bool               output = true;
  if (output)
    {
      deallog << "\n" << std::endl;
      deallog << "Weak form (ascii):\n"
              << assembler.as_ascii(decorator) << std::endl;
      deallog << "Weak form (LaTeX):\n"
              << assembler.as_latex(decorator) << std::endl;
      deallog << "\n" << std::endl;
      output = false;
    }

  // Now we pass in concrete objects to get data from
  // and assemble into.
  assembler.assemble_system(this->system_matrix,
                            this->system_rhs,
                            this->solution,
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
