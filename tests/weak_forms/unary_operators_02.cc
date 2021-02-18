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
// - Functors

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/symbolic_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim> fe(1);
  const QGauss<spacedim>    qf_cell(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const UpdateFlags       update_flags = update_quadrature_points;
  FEValues<dim, spacedim> fe_values(fe, qf_cell, update_flags);
  fe_values.reinit(dof_handler.begin_active());

  const unsigned int q_point = 0;

  {
    const std::string title = "Scalar functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ScalarFunctor coeff("c", "c");
    const auto          functor =
      value<double, dim, spacedim>(coeff,
                                   [](const FEValuesBase<dim, spacedim> &,
                                      const unsigned int) { return 1.0; });

    std::cout << "Value: "
              << (functor.template operator()<NumberType>(fe_values))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Tensor functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const TensorFunctor<2, spacedim> coeff("C", "C");
    const auto                       functor = value<double, spacedim>(
      coeff, [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        return Tensor<2, dim, double>(unit_symmetric_tensor<spacedim>());
      });

    std::cout << "Value: "
              << (functor.template operator()<NumberType>(fe_values))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Scalar function functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const Functions::ConstantFunction<spacedim, double>
                                          constant_scalar_function(1.0);
    const ScalarFunctionFunctor<spacedim> coeff("c", "c");
    const auto functor = value(coeff, constant_scalar_function);

    std::cout << "Value: "
              << (functor.template operator()<NumberType>(fe_values))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Tensor function functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ConstantTensorFunction<2, dim, double> constant_tensor_function(
      unit_symmetric_tensor<dim>());
    const TensorFunctionFunctor<2, spacedim> coeff("C", "C");
    const auto functor = value(coeff, constant_tensor_function);

    std::cout << "Value: "
              << (functor.template operator()<NumberType>(fe_values))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

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
