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


#include <deal.II/weakforms/functors.h>
// #include <deal.II/weakforms/spaces.h>
#include <deal.II/weakforms/symbolic_info.h>
#include <deal.II/weakforms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicNamesAscii naming_ascii;
  const SymbolicNamesLaTeX naming_latex;

  const ScalarFunctor<NumberType> scalar ("s", "s", naming_ascii, naming_latex);
  const VectorFunctor<dim,NumberType> vector ("v", "v", naming_ascii, naming_latex);
  const TensorFunctor<2,dim,NumberType> tensor2 ("T2", "T", naming_ascii, naming_latex);
  const TensorFunctor<3,dim,NumberType> tensor3 ("T3", "P", naming_ascii, naming_latex);
  const TensorFunctor<4,dim,NumberType> tensor4 ("T4", "K", naming_ascii, naming_latex);
  const SymmetricTensorFunctor<2,dim,NumberType> symm_tensor2 ("S2", "T", naming_ascii, naming_latex);
  const SymmetricTensorFunctor<4,dim,NumberType> symm_tensor4 ("S4", "K", naming_ascii, naming_latex);

  const auto s = value(scalar,[](const unsigned int){return 1.0;});
  const auto v = value(vector,[](const unsigned int){return Tensor<1,dim,NumberType>();});
  const auto T2 = value(tensor2,[](const unsigned int){return Tensor<2,dim,NumberType>();});
  const auto T3 = value(tensor3,[](const unsigned int){return Tensor<3,dim,NumberType>();});
  const auto T4 = value(tensor4,[](const unsigned int){return Tensor<4,dim,NumberType>();});
  const auto S2 = value(tensor2,[](const unsigned int){return SymmetricTensor<2,dim,NumberType>();});
  const auto S4 = value(tensor4,[](const unsigned int){return SymmetricTensor<4,dim,NumberType>();});

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "FUNCTOR CREATION" << std::endl;
    deallog << "Scalar: " << scalar.as_ascii() << std::endl;
    deallog << "Vector: " << vector.as_ascii() << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_ascii() << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_ascii() << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_ascii() << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_ascii() << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_ascii() << std::endl;

    deallog << "FUNCTOR VALUE SETTING" << std::endl;
    deallog << "Scalar: " << s.as_ascii() << " ; val: " << s(0) << std::endl;
    deallog << "Vector: " << v.as_ascii() << " ; val: " << v(0) << std::endl;
    deallog << "Tensor (rank 2): " << T2.as_ascii() << " ; val: " << T2(0) << std::endl;
    deallog << "Tensor (rank 3): " << T3.as_ascii() << " ; val: " << T3(0) << std::endl;
    deallog << "Tensor (rank 4): " << T4.as_ascii() << " ; val: " << T4(0) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << S2.as_ascii() << " ; val: " << S2(0) << std::endl;
    deallog << "SymmetricTensor (rank 4): " << S4.as_ascii() << " ; val: " << S4(0) << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "FUNCTOR CREATION" << std::endl;
    deallog << "Scalar: " << scalar.as_latex() << std::endl;
    deallog << "Vector: " << vector.as_latex() << std::endl;
    deallog << "Tensor (rank 2): " << tensor2.as_latex() << std::endl;
    deallog << "Tensor (rank 3): " << tensor3.as_latex() << std::endl;
    deallog << "Tensor (rank 4): " << tensor4.as_latex() << std::endl;
    deallog << "SymmetricTensor (rank 2): " << symm_tensor2.as_latex() << std::endl;
    deallog << "SymmetricTensor (rank 4): " << symm_tensor4.as_latex() << std::endl;

    deallog << "FUNCTOR VALUE SETTING" << std::endl;
    deallog << "Scalar: " << s.as_latex() << " ; val: " << s(0) << std::endl;
    deallog << "Vector: " << v.as_latex() << " ; val: " << v(0) << std::endl;
    deallog << "Tensor (rank 2): " << T2.as_latex() << " ; val: " << T2(0) << std::endl;
    deallog << "Tensor (rank 3): " << T3.as_latex() << " ; val: " << T3(0) << std::endl;
    deallog << "Tensor (rank 4): " << T4.as_latex() << " ; val: " << T4(0) << std::endl;
    deallog << "SymmetricTensor (rank 2): " << S2.as_latex() << " ; val: " << S2(0) << std::endl;
    deallog << "SymmetricTensor (rank 4): " << S4.as_latex() << " ; val: " << S4(0) << std::endl;

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
