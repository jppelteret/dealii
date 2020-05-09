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


// Check functor form stringization and printing
// - Functors

#include <deal.II/base/function_lib.h>

#include <deal.II/weakforms/functors.h>
// #include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/symbolic_decorations.h>
#include <deal.II/weakforms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const ScalarFunctor<NumberType> scalar ("s", "s", decorator);
  const VectorFunctor<dim,NumberType> vector ("v", "v", decorator);
  const TensorFunctor<2,dim,NumberType> tensor2 ("T2", "T", decorator);
  const TensorFunctor<3,dim,NumberType> tensor3 ("T3", "P", decorator);
  const TensorFunctor<4,dim,NumberType> tensor4 ("T4", "K", decorator);
  const SymmetricTensorFunctor<2,dim,NumberType> symm_tensor2 ("S2", "T", decorator);
  const SymmetricTensorFunctor<4,dim,NumberType> symm_tensor4 ("S4", "K", decorator);

  const auto s = value(scalar,[](const unsigned int){return 1.0;});
  const auto v = value(vector,[](const unsigned int){return Tensor<1,dim,NumberType>();});
  const auto T2 = value(tensor2,[](const unsigned int){return Tensor<2,dim,NumberType>();});
  const auto T3 = value(tensor3,[](const unsigned int){return Tensor<3,dim,NumberType>();});
  const auto T4 = value(tensor4,[](const unsigned int){return Tensor<4,dim,NumberType>();});
  const auto S2 = value(tensor2,[](const unsigned int){return SymmetricTensor<2,dim,NumberType>();});
  const auto S4 = value(tensor4,[](const unsigned int){return SymmetricTensor<4,dim,NumberType>();});

  const ScalarFunctionFunctor<dim,NumberType> scalar_func ("sf", "s", decorator);
  const TensorFunctionFunctor<2,dim,NumberType> tensor_func2 ("Tf2", "T", decorator);

  const Functions::ConstantFunction<dim,NumberType> constant_function(1);
  const ConstantTensorFunction<2,dim,NumberType> constant_tensor_function(unit_symmetric_tensor<dim>());
  const auto sf = value(scalar_func,constant_function);
  const auto T2f = value(tensor_func2,constant_tensor_function);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Scalar: " << scalar.as_ascii() << std::endl;
    deallog << "Vector: " << vector.as_ascii() << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_ascii() << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_ascii() << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_ascii() << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_ascii() << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_ascii() << std::endl;

    deallog << "Scalar function: " << scalar_func.as_ascii() << std::endl;
    deallog << "Tensor function (rank 2): " << tensor_func2.as_ascii() << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Scalar: " << scalar.as_latex() << std::endl;
    deallog << "Vector: " << vector.as_latex() << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_latex() << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_latex() << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_latex() << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_latex() << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_latex() << std::endl;

    deallog << "Scalar function: " << scalar_func.as_latex() << std::endl;
    deallog << "Tensor function (rank 2): " << tensor_func2.as_latex() << std::endl;

    deallog << std::endl;
  }  
  
  // Test values
  {
    LogStream::Prefix prefix("values");

    deallog << "Scalar: " << s(0) << std::endl;
    deallog << "Vector: " << v(0) << std::endl;
    deallog << "Tensor (rank 2): " << T2(0) << std::endl;
    deallog << "Tensor (rank 3): " << T3(0) << std::endl;
    deallog << "Tensor (rank 4): " << T4(0) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << S2(0) << std::endl;
    deallog << "SymmetricTensor (rank 4): " << S4(0) << std::endl;

    deallog << "Scalarfunction : " << sf(Point<dim>()) << std::endl;
    deallog << "Tensor function (rank 2): " << T2f(Point<dim>()) << std::endl;

    deallog << std::endl;
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
