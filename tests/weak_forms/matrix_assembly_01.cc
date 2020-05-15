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


// Check that integrator works for volumes and boundaries

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/meshworker/scratch_data.h>

// #include <deal.II/weak_forms/bilinear_forms.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/unary_operators.h>
// #include <deal.II/weak_forms/integrator.h>

#include "../tests.h"


template <int dim, int spacedim = dim>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));

  const FE_Q<dim, spacedim>  fe(1);
  const QGauss<spacedim>     qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1> qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 4, 0.0, 1.0);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix_std;
  SparseMatrix<double> system_matrix_wf_exemplar;
  SparseMatrix<double> system_matrix_wf;

  const UpdateFlags update_flags = update_values | update_JxW_values;

  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    sparsity_pattern.copy_from(dsp);

    system_matrix_std.reinit(sparsity_pattern);
    system_matrix_wf_exemplar.reinit(sparsity_pattern);
    system_matrix_wf.reinit(sparsity_pattern);
  }

  auto verify_assembly =
    [&system_matrix_std, &system_matrix_wf_exemplar, &system_matrix_wf]() {
      constexpr double tol = 1e-12;

      for (auto it1 = system_matrix_std.begin(),
                it2 = system_matrix_wf_exemplar.begin(),
                it3 = system_matrix_wf.begin();
           it1 != system_matrix_std.end();
           ++it1, ++it2, ++it3)
        {
          Assert(it2 != system_matrix_wf_exemplar.end(), ExcInternalError());
          Assert(it3 != system_matrix_wf.end(), ExcInternalError());

          AssertThrow(std::abs(it1->value() - it2->value()) < tol,
                      ExcMessage("Matrix entries are different (exemplar)."));

          AssertThrow(std::abs(it1->value() - it3->value()) < tol,
                      ExcMessage("Matrix entries are different (weak form)."));
        }
    };

  {
    std::cout << "Standard assembly" << std::endl;
    system_matrix_std = 0;

    FEValues<dim, spacedim> fe_values(fe, qf_cell, update_flags);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                   fe_values.shape_value(j, q) *
                                   fe_values.JxW(q);


        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix_std);
      }

    system_matrix_std.print(std::cout);
  }

  {
    using namespace WeakForms;

    std::cout << "Exemplar weak form assembly" << std::endl;
    system_matrix_wf_exemplar = 0;

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    MeshWorker::ScratchData<dim, spacedim> scratch_data(fe,
                                                        qf_cell,
                                                        update_flags);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.reinit(cell);

        const std::vector<double> &      JxW = fe_values.get_JxW_values();
        std::vector<std::vector<double>> Nx(fe_values.dofs_per_cell,
                                            std::vector<double>(
                                              fe_values.n_quadrature_points));
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            Nx[i][q] = fe_values.shape_value(i, q);

        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            for (const unsigned int q : fe_values.quadrature_point_indices())
              cell_matrix(i, j) += Nx[i][q] * Nx[j][q] * JxW[q];


        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix_wf_exemplar);
      }

    std::cout << "Exemplar weak form assembly" << std::endl;
    system_matrix_wf_exemplar.print(std::cout);
  }

  {
    using namespace WeakForms;

    std::cout << "Weak form assembly" << std::endl;
    system_matrix_wf = 0;

    // Customise the naming convensions, if we wish to.
    const SymbolicDecorations decorator;

    const TestFunction<dim, spacedim>  test(decorator);
    const TrialSolution<dim, spacedim> trial(decorator);

    const auto test_val  = value(test);
    const auto trial_val = value(trial);

    // const auto a_uv = bilinear_form(test, trial);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    MeshWorker::ScratchData<dim, spacedim> scratch_data(fe,
                                                        qf_cell,
                                                        update_flags);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.reinit(cell);

        const std::vector<double> &      JxW = fe_values.get_JxW_values();
        std::vector<std::vector<double>> Nx_test(
          fe_values.dofs_per_cell,
          std::vector<double>(fe_values.n_quadrature_points));
        std::vector<std::vector<double>> Nx_trial(
          fe_values.dofs_per_cell,
          std::vector<double>(fe_values.n_quadrature_points));
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              Nx_test[i][q] =
                test_val.template operator()<double>(fe_values, i, q);
              Nx_trial[i][q] =
                trial_val.template operator()<double>(fe_values, i, q);
            }

        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            for (const unsigned int q : fe_values.quadrature_point_indices())
              cell_matrix(i, j) += Nx_test[i][q] * Nx_trial[j][q] * JxW[q];


        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix_wf);
      }

    std::cout << "Weak form assembly" << std::endl;
    system_matrix_wf_exemplar.print(std::cout);
  }

  verify_assembly();

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
