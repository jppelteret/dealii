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

// Laplace problem: Assembly using self-linearizing weak form
// This test replicates step-6, but with a constant coefficient of unity.

// #include <deal.II/differentiation/ad.h>

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

  // using ADEnergyFunctional =
  //   AutoDifferentiation::EnergyFunctional<dim, ad_typecode>;

  constexpr auto ad_typecode =
    Differentiation::AD::NumberTypes::sacado_dfad_dfad;
  // using ADNumber_t = typename ADEnergyFunctional::ad_type;

  constexpr int spacedim = dim;

  // Symbolic types for test function, trial solution and a coefficient.
  const TestFunction<dim> test;
  // const TrialSolution<dim> trial;
  const FieldSolution<dim> solution;
  const ScalarFunctor      energy("e", "\\Psi");
  const ScalarFunctor      rhs_coeff("s", "s");

  const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s", "s");

  // const auto test_val   = test.value();
  // const auto test_grad  = test.gradient();
  // const auto trial_grad = trial.gradient();

  const auto test_ss  = test[subspace_extractor];
  const auto test_val = test_ss.value();

  const auto soln_ss   = solution[subspace_extractor];
  const auto soln_val  = soln_ss.value();    // Solution value
  const auto soln_grad = soln_ss.gradient(); // Solution gradient

  using ADNumber_t =
    typename Differentiation::AD::NumberTraits<double, ad_typecode>::ad_type;
  const auto energy_func = value<ad_typecode, dim, spacedim>(
    energy, [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
      return ADNumber_t(0.0);
    });
  const auto rhs_coeff_func =
    value<double, dim, spacedim>(rhs_coeff,
                                 [](const FEValuesBase<dim, spacedim> &,
                                    const unsigned int) { return 1.0; });

  // using Functor_t = std::function<ADNumber_t(Tensor<1, dim, ADNumber_t>)>;
  // auto f          = [](const Tensor<1, dim, ADNumber_t> &grad_u) ->
  // ADNumber_t {
  //   return ADNumber_t(0.0);
  // };
  // const ADEnergyFunctional energy_functional(f, soln_grad);

  // struct Functor
  // {

  // } f;


  // const VectorFunctor<dim> v_coeff("v", "v");
  // const auto f = v_coeff.template value<double>([](const FEValuesBase<dim,
  // spacedim> &,
  //                                   const unsigned int) { return
  //                                   Tensor<1,dim>{}; });

  // const auto f =
  //   value<double, spacedim>(v_coeff,
  //                                [](const FEValuesBase<dim, spacedim> &,
  //                                   const unsigned int) { Tensor<1,dim>{};
  //                                   });

  // const VectorFunctor<dim> v_coeff("v", "v");
  // const auto               f = value<double, spacedim>(
  //   v_coeff, [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
  //     return Tensor<1, dim> ();
  //   });

  MatrixBasedAssembler<dim> assembler;
  assembler +=
    self_linearizing_energy_functional_form(energy_func, soln_grad).dV();

  // assembler += ad_energy_functional_form<dim, ad_typecode>(f,
  // soln_grad).dV();

  // assembler += internal::linearized_form(
  //   {[](const Tensor<1,spacedim> &/*grad_u*/){
  //   return Tensor<1, spacedim>{};}},
  //   {[](const Tensor<2,spacedim> &/*grad_u*/){
  //     return 1.0 * unit_symmetric_tensor<>(spacedim);},
  //   soln_grad); // LHS contribution
  //   // assembler += bilinear_form(test_grad, mat_coeff_func, trial_grad)
  //   //                .dV();                                    // LHS
  //   //                contribution

  //   assembler +=
  //     internal::linearized_form({[](const double & /*u*/) { return 1.0; }},
  //                               {[](const double & /*u*/) { return 0.0; }},
  //                               soln_val);                   // RHS
  //                               contribution
  assembler -= linear_form(test_val, rhs_coeff_func).dV(); // RHS contribution

  // Look at what we're going to compute
  const SymbolicDecorations decorator;
  static bool               output = true;
  if (output)
    {
      std::cout << "Weak form (ascii):\n"
                << assembler.as_ascii(decorator) << std::endl;
      std::cout << "Weak form (LaTeX):\n"
                << assembler.as_latex(decorator) << std::endl;
      output = false;
    }

  throw;

  //   // Compute the residual, linearisations etc. using the energy form
  //   assembler.update_solution(this->solution, this->dof_handler,
  //   this->qf_cell);

  //   // Now we pass in concrete objects to get data from
  //   // and assemble into.
  //   assembler.assemble_system(this->system_matrix,
  //                             this->system_rhs,
  //                             this->constraints,
  //                             this->dof_handler,
  //                             this->qf_cell);
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
