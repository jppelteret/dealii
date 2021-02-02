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

// Elasticity problem: Assembly using self-linearizing weak form in conjunction
// with automatic differentiation.
// This test replicates step-8 exactly.

#include <deal.II/base/function.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/weak_forms/weak_forms.h>

#include "../tests.h"

#include "wf_common_tests/step-8.h"


using namespace dealii;



template <int dim>
class Step8 : public Step8_Base<dim>
{
public:
  Step8();

protected:
  void
  assemble_system() override;
};


template <int dim>
Step8<dim>::Step8()
  : Step8_Base<dim>()
{}


template <int dim>
void
Step8<dim>::assemble_system()
{
  using namespace WeakForms;
  using namespace Differentiation;

  constexpr int  spacedim = dim;
  constexpr auto ad_typecode =
    Differentiation::AD::NumberTypes::sacado_dfad_dfad;
  using ADNumber_t =
    typename Differentiation::AD::NumberTraits<double, ad_typecode>::ad_type;

  // Symbolic types for test function, and a coefficient.
  const TestFunction<dim>          test;
  const FieldSolution<dim>         solution;
  const SubSpaceExtractors::Vector subspace_extractor(0, "u", "\\mathbf{u}");

  // const TensorFunctionFunctor<4, dim> mat_coeff("C", "\\mathcal{C}");
  const VectorFunctionFunctor<dim> rhs_coeff("s", "\\mathbf{s}");
  const Coefficient<dim>           coefficient;
  const RightHandSide<dim>         rhs;

  const auto test_ss = test[subspace_extractor];
  const auto soln_ss = solution[subspace_extractor];

  const auto test_val  = test_ss.value();
  const auto soln_grad = soln_ss.gradient();

  const auto energy_func = energy_functor("e", "\\Psi", soln_grad);
  using EnergyADNumber_t =
    typename decltype(energy_func)::template ad_type<double, ad_typecode>;
  static_assert(std::is_same<ADNumber_t, EnergyADNumber_t>::value,
                "Expected identical AD number types");

  const auto energy = energy_func.template value<ADNumber_t, dim, spacedim>(
    [&coefficient](const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                   const std::vector<std::string> &              solution_names,
                   const unsigned int                            q_point,
                   const Tensor<2, spacedim, ADNumber_t> &       grad_u) {
      // Sacado is unbelievably annoying. If we don't explicitly
      // cast this return type then we get a segfault.
      // i.e. don't return the result inline!
      const Point<spacedim> &p = scratch_data.get_quadrature_points()[q_point];
      const auto             C = coefficient.value(p);
      const ADNumber_t       energy = 0.5 * contract3(grad_u, C, grad_u);
      return energy;
    },
    UpdateFlags::update_quadrature_points);

  MatrixBasedAssembler<dim> assembler;
  assembler += energy_functional_form(energy, soln_grad).dV() -
               linear_form(test_val, rhs_coeff(rhs)).dV();

  // Look at what we're going to compute
  const SymbolicDecorations decorator;
  static bool               output = true;
  if (output)
    {
      deallog << "Weak form (ascii):\n"
              << assembler.as_ascii(decorator) << std::endl;
      deallog << "Weak form (LaTeX):\n"
              << assembler.as_latex(decorator) << std::endl;
      output = false;
    }

  // Now we pass in concrete objects to get data from
  // and assemble into.
  const QGauss<dim> qf_cell(this->fe.degree + 1);
  assembler.assemble_system(this->system_matrix,
                            this->system_rhs,
                            this->solution,
                            this->constraints,
                            this->dof_handler,
                            qf_cell);
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
      Step8<2> elastic_problem_2d;
      elastic_problem_2d.run();
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
