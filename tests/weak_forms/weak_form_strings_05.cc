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
// - Integrals

#include <deal.II/base/function_lib.h>

#include <deal.II/weak_forms/bilinear_forms.h>
#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/integral.h>
#include <deal.II/weak_forms/linear_forms.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const ScalarFunctor                 scalar("s", "s", decorator);
  const ScalarFunctionFunctor<dim>    scalar_func("sf", "s", decorator);
  const TensorFunctionFunctor<2, dim> tensor_func2("Tf2", "T", decorator);

  const TestFunction<dim, spacedim>  test(decorator);
  const TrialSolution<dim, spacedim> trial(decorator);
  const FieldSolution<dim, spacedim> soln(decorator);

  const VolumeIntegral    integral_dV(decorator);
  const BoundaryIntegral  integral_dA(decorator);
  const InterfaceIntegral integral_dI(decorator);

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "Volume integral: " << integral_dV.as_ascii() << std::endl;
    deallog << "Boundary integral: " << integral_dA.as_ascii() << std::endl;
    deallog << "Interface integral: " << integral_dI.as_ascii() << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "Volume integral: " << integral_dV.as_latex() << std::endl;
    deallog << "Boundary integral: " << integral_dA.as_latex() << std::endl;
    deallog << "Interface integral: " << integral_dI.as_latex() << std::endl;

    deallog << std::endl;
  }

  const auto s =
    value<NumberType>(scalar, [](const unsigned int) { return 1.0; });

  const Functions::ConstantFunction<dim, NumberType> constant_scalar_function(
    1);
  const ConstantTensorFunction<2, dim, NumberType> constant_tensor_function(
    unit_symmetric_tensor<dim>());
  const auto sf  = value(scalar_func, constant_scalar_function);
  const auto T2f = value(tensor_func2, constant_tensor_function);

  const auto l_form = linear_form(test, soln); // Note: Not really permissible
  const auto bl_form =
    bilinear_form(test, soln, trial); // Note: Not really permissible

  const auto s_dV   = value(integral_dV, s);
  const auto T2f_dA = value(integral_dA, T2f);
  const auto sf_dI  = integrate(sf, integral_dI);

  const auto blf_dV = integrate(bl_form, integral_dV);
  const auto blf_dA = integrate(bl_form, integral_dA);
  const auto blf_dI = integrate(bl_form, integral_dI);

  const auto lf_dV = integrate(l_form, integral_dV);
  const auto lf_dA = integrate(l_form, integral_dA);
  const auto lf_dI = integrate(l_form, integral_dI);

  // Test values
  {
    LogStream::Prefix prefix("values");

    deallog << "Volume integral: " << s_dV.as_latex() << std::endl;
    deallog << "Boundary integral: " << T2f_dA.as_latex() << std::endl;
    deallog << "Interface integral: " << sf_dI.as_latex() << std::endl;

    deallog << "Integrate function: " << std::endl;
    deallog << "Bilinear form (Volume integral): " << blf_dV.as_latex()
            << std::endl;
    deallog << "Bilinear form (Boundary integral): " << blf_dA.as_latex()
            << std::endl;
    deallog << "Bilinear form (Interface integral): " << blf_dI.as_latex()
            << std::endl;

    deallog << "Linear form (Volume integral: " << lf_dV.as_latex()
            << std::endl;
    deallog << "Linear form (Boundary integral: " << lf_dA.as_latex()
            << std::endl;
    deallog << "Linear form (Interface integral: " << lf_dI.as_latex()
            << std::endl;

    deallog << "Form integral: " << std::endl;
    deallog << "Bilinear form (Volume integral): " << bl_form.dV().as_latex()
            << std::endl;
    deallog << "Bilinear form (Boundary integral): " << bl_form.dA().as_latex()
            << std::endl;
    deallog << "Bilinear form (Interface integral): " << bl_form.dI().as_latex()
            << std::endl;

    deallog << "Linear form (Volume integral: " << l_form.dV().as_latex()
            << std::endl;
    deallog << "Linear form (Boundary integral: " << l_form.dA().as_latex()
            << std::endl;
    deallog << "Linear form (Interface integral: " << l_form.dI().as_latex()
            << std::endl;

    deallog << "Form integral with subregions: " << std::endl;
    deallog << "Bilinear form (Volume integral): " << bl_form.dV({1,2,3}).as_latex()
            << std::endl;
    deallog << "Bilinear form (Boundary integral): " << bl_form.dA({4,5,6}).as_latex()
            << std::endl;
    deallog << "Bilinear form (Interface integral): " << bl_form.dI({7,8,9}).as_latex()
            << std::endl;

    deallog << "Linear form (Volume integral: " << l_form.dV({1,2,3}).as_latex()
            << std::endl;
    deallog << "Linear form (Boundary integral: " << l_form.dA({4,5,6}).as_latex()
            << std::endl;
    deallog << "Linear form (Interface integral: " << l_form.dI({7,8,9}).as_latex()
            << std::endl;

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
