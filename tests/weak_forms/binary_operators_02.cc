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


// Check that binary operators work
// - Field solution

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/weak_forms/binary_operators.h>
#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim>  fe(1);
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

  const UpdateFlags update_flags_cell =
    update_quadrature_points | update_values | update_gradients |
    update_hessians | update_3rd_derivatives;
  const UpdateFlags           update_flags_face = update_normal_vectors;
  FEValues<dim, spacedim>     fe_values(fe, qf_cell, update_flags_cell);
  FEFaceValues<dim, spacedim> fe_face_values(fe, qf_face, update_flags_face);

  const auto cell = dof_handler.begin_active();
  fe_values.reinit(cell);
  fe_face_values.reinit(cell, 0);

  const unsigned int q_point = 0;

  {
    const std::string title = "Scalar";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ScalarFunctor c1("c1", "c1");
    const auto          f1 =
      value<double, dim, spacedim>(c1,
                                   [](const FEValuesBase<dim, spacedim> &,
                                      const unsigned int) { return 2.0; });

    const FieldSolution<dim, spacedim> field_solution;
    const auto                         value     = field_solution.value();
    const auto                         gradient  = field_solution.gradient();
    const auto                         laplacian = field_solution.laplacian();
    const auto                         hessian   = field_solution.hessian();
    const auto third_derivative = field_solution.third_derivative();

    // TODO: This does not work because we now work with local solution
    // values, not the global vector

    std::cout << "Scalar * value: "
              << ((f1 * value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Scalar * gradient: "
              << ((f1 * gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Scalar * Laplacian: "
              << ((f1 * laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Scalar * Hessian: "
              << ((f1 * hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Scalar * third derivative: "
              << ((f1 * third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "Scalar + value: "
              << ((f1 + value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Scalar - value: "
              << ((f1 - value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "Scalar + Laplacian: "
              << ((f1 + laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Scalar - Laplacian: "
              << ((f1 - laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Vector";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const VectorFunctor<dim> v1("v1", "v1");
    const auto               f1 = value<double, spacedim>(
      v1, [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        Tensor<1, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        return t;
      });

    const FieldSolution<dim, spacedim> field_solution;
    const auto                         value     = field_solution.value();
    const auto                         gradient  = field_solution.gradient();
    const auto                         laplacian = field_solution.laplacian();
    const auto                         hessian   = field_solution.hessian();
    const auto third_derivative = field_solution.third_derivative();

    std::cout << "Vector * value: "
              << ((f1 * value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Vector * gradient: "
              << ((f1 * gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Vector * Laplacian: "
              << ((f1 * laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Vector * Hessian: "
              << ((f1 * hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Vector * third derivative: "
              << ((f1 * third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "Vector + gradient: "
              << ((f1 + gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Vector - gradient: "
              << ((f1 - gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Tensor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const TensorFunctor<2, dim> T1("T1", "T1");
    const auto                  f1 = value<double, spacedim>(
      T1, [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        Tensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        return t;
      });

    const FieldSolution<dim, spacedim> field_solution;
    const auto                         value     = field_solution.value();
    const auto                         gradient  = field_solution.gradient();
    const auto                         laplacian = field_solution.laplacian();
    const auto                         hessian   = field_solution.hessian();
    const auto third_derivative = field_solution.third_derivative();

    std::cout << "Tensor * value: "
              << ((f1 * value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Tensor * gradient: "
              << ((f1 * gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Tensor * Laplacian: "
              << ((f1 * laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Tensor * Hessian: "
              << ((f1 * hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Tensor * third derivative: "
              << ((f1 * third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "Tensor + Hessian: "
              << ((f1 + hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Tensor - Hessian: "
              << ((f1 - hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "SymmetricTensor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const SymmetricTensorFunctor<2, dim> S1("S1", "S1");
    const auto                           f1 = value<double, spacedim>(
      S1, [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        SymmetricTensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        return t;
      });

    const FieldSolution<dim, spacedim> field_solution;
    const auto                         value     = field_solution.value();
    const auto                         gradient  = field_solution.gradient();
    const auto                         laplacian = field_solution.laplacian();
    const auto                         hessian   = field_solution.hessian();
    const auto third_derivative = field_solution.third_derivative();

    std::cout << "SymmetricTensor * value: "
              << ((f1 * value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "SymmetricTensor * gradient: "
              << ((f1 * gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "SymmetricTensor * Laplacian: "
              << ((f1 * laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "SymmetricTensor * Hessian: "
              << ((f1 * hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "SymmetricTensor * third derivative: "
              << ((f1 * third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "SymmetricTensor + Hessian: "
              << ((f1 + hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "SymmetricTensor - Hessian: "
              << ((f1 - hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Field solution";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const FieldSolution<dim, spacedim> field_solution;
    const auto                         value     = field_solution.value();
    const auto                         gradient  = field_solution.gradient();
    const auto                         laplacian = field_solution.laplacian();
    const auto                         hessian   = field_solution.hessian();
    const auto third_derivative = field_solution.third_derivative();

    std::cout << "value + value: "
              << ((value + value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "gradient + gradient: "
              << ((gradient + gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Laplacian + Laplacian: "
              << ((laplacian + laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Hessian + Hessian: "
              << ((hessian + hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "third derivative + third derivative: "
              << ((third_derivative + third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "value - value: "
              << ((value - value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "gradient - gradient: "
              << ((gradient - gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Laplacian - Laplacian: "
              << ((laplacian - laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Hessian - Hessian: "
              << ((hessian - hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "third derivative - third derivative: "
              << ((third_derivative - third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;

    std::cout << "value * value: "
              << ((value * value)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "gradient * gradient: "
              << ((gradient * gradient)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Laplacian * Laplacian: "
              << ((laplacian * laplacian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "Hessian * Hessian: "
              << ((hessian * hessian)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
              << std::endl;
    std::cout << "third derivative * third derivative: "
              << ((third_derivative * third_derivative)
                    .template operator()<NumberType>(fe_values,
                                                     solution))[q_point]
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
