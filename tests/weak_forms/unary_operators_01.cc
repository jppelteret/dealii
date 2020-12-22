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


// Check that unary operators work
// - Trial function, test function, field solution (scalar-valued finite
// element)

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim>  fe(3);
  const QGauss<spacedim>     qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1> qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,
                           Functions::CosineFunction<spacedim>(
                             fe.n_components()),
                           solution);

  const UpdateFlags update_flags =
    update_values | update_gradients | update_hessians | update_3rd_derivatives;
  FEValues<dim, spacedim> fe_values(fe, qf_cell, update_flags);

  const auto cell = dof_handler.begin_active();
  std::vector<double> local_dof_values(fe.dofs_per_cell);
  cell->get_dof_values(solution,
                       local_dof_values.begin(),
                       local_dof_values.end());

  fe_values.reinit(cell);
  const unsigned int q_point   = 0;
  const unsigned int dof_index = 0;

  {
    const std::string title = "Test function";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;
    const TestFunction<dim, spacedim> test;

    std::cout << "Value: "
              << (test.value().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Gradient: "
              << (test.gradient().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Laplacian: "
              << (test.laplacian().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Hessian: "
              << (test.hessian().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Third derivative: "
              << (test.third_derivative().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Trial solution";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;
    const TrialSolution<dim, spacedim> trial;

    std::cout << "Value: "
              << (trial.value().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Gradient: "
              << (trial.gradient().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Laplacian: "
              << (trial.laplacian().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Hessian: "
              << (trial.hessian().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;
    std::cout << "Third derivative: "
              << (trial.third_derivative().template operator()<NumberType>(
                   fe_values, dof_index, q_point))
              << std::endl;

    deallog << "OK" << std::endl;
  }

// Not implemented
//   {
//     const std::string title = "Field solution";
//     std::cout << title << std::endl;
//     deallog << title << std::endl;

//     using namespace WeakForms;
//     const FieldSolution<dim, spacedim> field_solution;

//     std::cout << "Value: "
//               << (field_solution.value().template operator()<NumberType>(
//                    fe_values, local_dof_values))[q_point]
//               << std::endl;
//     std::cout << "Gradient: "
//               << (field_solution.gradient().template operator()<NumberType>(
//                    fe_values, local_dof_values))[q_point]
//               << std::endl;
//     std::cout << "Laplacian: "
//               << (field_solution.laplacian().template operator()<NumberType>(
//                    fe_values, local_dof_values))[q_point]
//               << std::endl;
//     std::cout << "Hessian: "
//               << (field_solution.hessian().template operator()<NumberType>(
//                    fe_values, local_dof_values))[q_point]
//               << std::endl;
//     std::cout << "Third derivative: "
//               << (field_solution.third_derivative().template
//                   operator()<NumberType>(fe_values, local_dof_values))[q_point]
//               << std::endl;

//     deallog << "OK" << std::endl;
//   }

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
