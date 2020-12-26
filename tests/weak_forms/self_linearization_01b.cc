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


// Check that (internal) UnaryOp conversion from a field solution
// to a test function or trial solution works as anticipated.
// - Sub-Space: Vector
//
// This test is derived from weak_form_strings_01b.cc


#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/weak_forms/self_linearizing_forms.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/unary_operators.h>

#include "../tests.h"


template <int dim>
struct RunCurl;

template <>
struct RunCurl<2>
{
  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_ascii(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {}

  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_latex(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {}
};

template <>
struct RunCurl<3>
{
  using Converter = WeakForms::SelfLinearization::internal::ConvertTo;

  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_ascii(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {
    using namespace WeakForms;

    deallog << "SPACE FUNCTIONS: Curl" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(curl(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(curl(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << curl(soln_ss).as_ascii(decorator) << std::endl;

    deallog << std::endl;
  }

  template <typename TestSSType, typename TrialSSType, typename SolutionSSType>
  static void
  as_latex(const TestSSType &                    test_ss,
           const TrialSSType &                   trial_ss,
           const SolutionSSType &                soln_ss,
           const WeakForms::SymbolicDecorations &decorator)
  {
    using namespace WeakForms;

    deallog << "SPACE FUNCTIONS: Curl" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(curl(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(curl(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << curl(soln_ss).as_latex(decorator) << std::endl;

    deallog << std::endl;
  }
};


template <int dim,
          int spacedim        = dim,
          typename NumberType = double,
          typename SubSpaceExtractorType>
void
run(const SubSpaceExtractorType &subspace_extractor)
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;
  using Converter = WeakForms::SelfLinearization::internal::ConvertTo;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const TestFunction<dim, spacedim>  test;
  const TrialSolution<dim, spacedim> trial;
  const FieldSolution<dim, spacedim> soln;

  const auto test_ss  = test[subspace_extractor];
  const auto trial_ss = trial[subspace_extractor];
  const auto soln_ss  = soln[subspace_extractor];

  // Test strings
  {
    LogStream::Prefix prefix("string");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test_ss.as_ascii(decorator) << std::endl;
    deallog << "Trial solution: " << trial_ss.as_ascii(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Value" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(value(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(value(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << value(soln_ss).as_ascii(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(gradient(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(gradient(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << gradient(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Symmetric gradient" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(symmetric_gradient(soln_ss))
                 .as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(symmetric_gradient(soln_ss))
                 .as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << symmetric_gradient(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Divergence" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(divergence(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog
      << "Trial solution: "
      << Converter::trial_solution(divergence(soln_ss)).as_ascii(decorator)
      << std::endl;
    deallog << "Solution: " << divergence(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    RunCurl<dim>::as_ascii(test_ss, trial_ss, soln_ss, decorator);

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(hessian(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(hessian(soln_ss)).as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << hessian(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog
      << "Test function: "
      << Converter::test_function(third_derivative(soln_ss)).as_ascii(decorator)
      << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(third_derivative(soln_ss))
                 .as_ascii(decorator)
            << std::endl;
    deallog << "Solution: " << third_derivative(soln_ss).as_ascii(decorator)
            << std::endl;

    deallog << std::endl;
  }

  // Test LaTeX
  {
    LogStream::Prefix prefix("LaTeX");

    deallog << "SPACE CREATION" << std::endl;
    deallog << "Test function: " << test_ss.as_latex(decorator) << std::endl;
    deallog << "Trial solution: " << trial_ss.as_latex(decorator) << std::endl;
    deallog << "Solution: " << soln_ss.as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Value" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(value(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(value(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << value(soln_ss).as_latex(decorator) << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Gradient" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(gradient(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(gradient(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << gradient(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Symmetric gradient" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(symmetric_gradient(soln_ss))
                 .as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(symmetric_gradient(soln_ss))
                 .as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << symmetric_gradient(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Divergence" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(divergence(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog
      << "Trial solution: "
      << Converter::trial_solution(divergence(soln_ss)).as_latex(decorator)
      << std::endl;
    deallog << "Solution: " << divergence(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    RunCurl<dim>::as_latex(test_ss, trial_ss, soln_ss, decorator);

    deallog << "SPACE FUNCTIONS: Hessian" << std::endl;
    deallog << "Test function: "
            << Converter::test_function(hessian(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(hessian(soln_ss)).as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << hessian(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;

    deallog << "SPACE FUNCTIONS: Third derivative" << std::endl;
    deallog
      << "Test function: "
      << Converter::test_function(third_derivative(soln_ss)).as_latex(decorator)
      << std::endl;
    deallog << "Trial solution: "
            << Converter::trial_solution(third_derivative(soln_ss))
                 .as_latex(decorator)
            << std::endl;
    deallog << "Solution: " << third_derivative(soln_ss).as_latex(decorator)
            << std::endl;

    deallog << std::endl;
  }
}


int
main()
{
  initlog();

  const WeakForms::SubSpaceExtractors::Vector subspace_extractor(0,
                                                                 "u",
                                                                 "\\mathbf{u}");
  run<2>(subspace_extractor);

  deallog << "OK" << std::endl;
}
