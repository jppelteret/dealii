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

// Finite strain elasticity problem: Assembly using weak forms
// This test replicates step-44 exactly.

#include <deal.II/weak_forms/weak_forms.h>

#include "../tests.h"

#include "wf_common_tests/step-44.h"

namespace Step44
{
  template <int dim>
  class Step44 : public Step44_Base<dim>
  {
  public:
    Step44(const std::string &input_file)
      : Step44_Base<dim>(input_file)
    {}

  protected:
    void
    assemble_system(const BlockVector<double> &solution_delta) override;
  };
  template <int dim>
  void
  Step44<dim>::assemble_system(const BlockVector<double> &solution_delta)
  {
    using namespace WeakForms;

    this->timer.enter_subsection("Assemble system");
    std::cout << " ASM_SYS " << std::flush;
    this->tangent_matrix = 0.0;
    this->system_rhs     = 0.0;
    const BlockVector<double> solution_total(
      this->get_total_solution(solution_delta));

    // const UpdateFlags uf_cell(update_values | update_gradients |
    //                           update_JxW_values);
    // const UpdateFlags uf_face(update_values | update_normal_vectors |
    //                           update_JxW_values);

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim>          test;
    const TrialSolution<dim>         trial;
    const SubSpaceExtractors::Vector subspace_extractor(0, "u", "\\mathbf{u}");

    const TensorFunctionFunctor<4, dim> mat_coeff("C", "\\mathcal{C}");
    const VectorFunctionFunctor<dim>    rhs_coeff("s", "\\mathbf{s}");
    const auto mat_coeff_func = mat_coeff(Coefficient<dim>());
    const auto rhs_coeff_func = rhs_coeff(RightHandSide<dim>());

    // // ERROR: PURE VIRTUAL FUNCTION CALLED - Need clone!
    const auto test_ss  = test[subspace_extractor];
    const auto trial_ss = trial[subspace_extractor];

    const auto test_val   = test_ss.value();
    const auto test_grad  = test_ss.gradient();
    const auto trial_grad = trial_ss.gradient();


    MatrixBasedAssembler<dim> assembler;
    // assembler += bilinear_form(test[subspace_extractor].gradient(),
    // mat_coeff_func, trial[subspace_extractor].gradient()).dV()
    //            - linear_form(test[subspace_extractor].value(),
    //            rhs_coeff_func).dV();
    assembler += bilinear_form(test_grad, mat_coeff_func, trial_grad).dV() -
                 linear_form(test_val, rhs_coeff_func).dV();

    // Look at what we're going to compute
    const SymbolicDecorations decorator;
    std::cout << "Weak form (ascii):\n"
              << assembler.as_ascii(decorator) << std::endl;
    std::cout << "Weak form (LaTeX):\n"
              << assembler.as_latex(decorator) << std::endl;

    // Now we pass in concrete objects to get data from
    // and assemble into.
    const QGauss<dim> qf_cell(this->fe.degree + 1);
    assembler.assemble_system(this->system_matrix,
                              this->system_rhs,
                              this->constraints,
                              this->dof_handler,
                              qf_cell);

    this->timer.leave_subsection();
  }
  // template <int dim>
  // void
  // Step44<dim>::assemble_system_one_cell(
  //   const typename DoFHandler<dim>::active_cell_iterator &cell,
  //   ScratchData_ASM &                                     scratch,
  //   PerTaskData_ASM &                                     data) const
  // {
  //   data.reset();
  //   scratch.reset();
  //   scratch.fe_values_ref.reinit(cell);
  //   cell->get_dof_indices(data.local_dof_indices);
  //   const std::vector<
  //     std::shared_ptr<const PointHistory<dim>>>
  //     lqph = this->quadrature_point_history.get_data(cell);
  //   Assert(lqph.size() == this->n_q_points, ExcInternalError());
  //   for (unsigned int q_point = 0; q_point < this->n_q_points; ++q_point)
  //     {
  //       const Tensor<2, dim> F_inv = lqph[q_point]->get_F_inv();
  //       for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
  //         {
  //           const unsigned int k_group =
  //           this->fe.system_to_base_index(k).first.first; if (k_group ==
  //           this->u_dof)
  //             {
  //               scratch.grad_Nx[q_point][k] =
  //                 scratch.fe_values_ref[this->u_fe].gradient(k, q_point) *
  //                 F_inv;
  //               scratch.symm_grad_Nx[q_point][k] =
  //                 symmetrize(scratch.grad_Nx[q_point][k]);
  //             }
  //           else if (k_group == this->p_dof)
  //             scratch.Nx[q_point][k] =
  //               scratch.fe_values_ref[this->p_fe].value(k, q_point);
  //           else if (k_group == this->J_dof)
  //             scratch.Nx[q_point][k] =
  //               scratch.fe_values_ref[this->J_fe].value(k, q_point);
  //           else
  //             Assert(k_group <= this->J_dof, ExcInternalError());
  //         }
  //     }
  //   for (unsigned int q_point = 0; q_point < this->n_q_points; ++q_point)
  //     {
  //       const SymmetricTensor<2, dim> tau    = lqph[q_point]->get_tau();
  //       const Tensor<2, dim>          tau_ns = lqph[q_point]->get_tau();
  //       const double                  J_tilde = lqph[q_point]->get_J_tilde();
  //       const double                  p_tilde = lqph[q_point]->get_p_tilde();
  //       const SymmetricTensor<4, dim> Jc  = lqph[q_point]->get_Jc();
  //       const double dPsi_vol_dJ          = lqph[q_point]->get_dPsi_vol_dJ();
  //       const double d2Psi_vol_dJ2        =
  //       lqph[q_point]->get_d2Psi_vol_dJ2(); const double det_F =
  //       lqph[q_point]->get_det_F(); const std::vector<double> & N =
  //       scratch.Nx[q_point]; const std::vector<SymmetricTensor<2, dim>>
  //       &symm_grad_Nx =
  //         scratch.symm_grad_Nx[q_point];
  //       const std::vector<Tensor<2, dim>> &grad_Nx =
  //       scratch.grad_Nx[q_point]; const double JxW =
  //       scratch.fe_values_ref.JxW(q_point); for (unsigned int i = 0; i <
  //       this->dofs_per_cell; ++i)
  //         {
  //           const unsigned int component_i =
  //             this->fe.system_to_component_index(i).first;
  //           const unsigned int i_group =
  //           this->fe.system_to_base_index(i).first.first;

  //           if (i_group == this->u_dof)
  //             data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
  //           else if (i_group == this->p_dof)
  //             data.cell_rhs(i) -= N[i] * (det_F - J_tilde) * JxW;
  //           else if (i_group == this->J_dof)
  //             data.cell_rhs(i) -= N[i] * (dPsi_vol_dJ - p_tilde) * JxW;
  //           else
  //             Assert(i_group <= this->J_dof, ExcInternalError());

  //           for (unsigned int j = 0; j <= i; ++j)
  //             {
  //               const unsigned int component_j =
  //                 this->fe.system_to_component_index(j).first;
  //               const unsigned int j_group =
  //                 this->fe.system_to_base_index(j).first.first;
  //               if ((i_group == j_group) && (i_group == this->u_dof))
  //                 {
  //                   data.cell_matrix(i, j) += symm_grad_Nx[i] *
  //                                             Jc // The material
  //                                             contribution:
  //                                             * symm_grad_Nx[j] * JxW;
  //                   if (component_i ==
  //                       component_j) // geometrical stress contribution
  //                     data.cell_matrix(i, j) += grad_Nx[i][component_i] *
  //                     tau_ns *
  //                                               grad_Nx[j][component_j] *
  //                                               JxW;
  //                 }
  //               else if ((i_group == this->p_dof) && (j_group ==
  //               this->u_dof))
  //                 {
  //                   data.cell_matrix(i, j) +=
  //                     N[i] * det_F *
  //                     (symm_grad_Nx[j] * StandardTensors<dim>::I) * JxW;
  //                 }
  //               else if ((i_group == this->J_dof) && (j_group ==
  //               this->p_dof))
  //                 data.cell_matrix(i, j) -= N[i] * N[j] * JxW;
  //               else if ((i_group == j_group) && (i_group == this->J_dof))
  //                 data.cell_matrix(i, j) += N[i] * d2Psi_vol_dJ2 * N[j] *
  //                 JxW;
  //               else
  //                 Assert((i_group <= this->J_dof) && (j_group <=
  //                 this->J_dof),
  //                        ExcInternalError());
  //             }
  //         }
  //     }
  //   for (const unsigned int face : GeometryInfo<dim>::face_indices())
  //     if (cell->face(face)->at_boundary() == true &&
  //         cell->face(face)->boundary_id() == 6)
  //       {
  //         scratch.fe_face_values_ref.reinit(cell, face);
  //         for (unsigned int f_q_point = 0; f_q_point < this->n_q_points_f;
  //              ++f_q_point)
  //           {
  //             const Tensor<1, dim> &N =
  //               scratch.fe_face_values_ref.normal_vector(f_q_point);
  //             static const double p0 =
  //               -4.0 / (this->parameters.scale * this->parameters.scale);
  //             const double         time_ramp = (this->time.current() /
  //             this->time.end()); const double         pressure  = p0 *
  //             this->parameters.p_p0 * time_ramp; const Tensor<1, dim>
  //             traction  = pressure * N; for (unsigned int i = 0; i <
  //             this->dofs_per_cell; ++i)
  //               {
  //                 const unsigned int i_group =
  //                   this->fe.system_to_base_index(i).first.first;
  //                 if (i_group == this->u_dof)
  //                   {
  //                     const unsigned int component_i =
  //                       this->fe.system_to_component_index(i).first;
  //                     const double Ni =
  //                       scratch.fe_face_values_ref.shape_value(i, f_q_point);
  //                     const double JxW =
  //                       scratch.fe_face_values_ref.JxW(f_q_point);
  //                     data.cell_rhs(i) += (Ni * traction[component_i]) * JxW;
  //                   }
  //               }
  //           }
  //       }


  //   for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
  //     for (unsigned int j = i + 1; j < this->dofs_per_cell; ++j)
  //       data.cell_matrix(i, j) = data.cell_matrix(j, i);
  // }
} // namespace Step44

int
main(int argc, char **argv)
{
  initlog();
  deallog << std::setprecision(9);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  using namespace dealii;
  try
    {
      const unsigned int dim = 3;
      Step44::Step44<dim>        solid(SOURCE_DIR "/prm/parameters-step-44.prm");
      solid.run();
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
  return 0;
}
