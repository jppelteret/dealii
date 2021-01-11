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

#include <deal.II/weak_forms/symbolic_decorations.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  SymbolicNames::SymbolicNames(
    const std::string solution_field,
    const std::string test_function,
    const std::string trial_solution,
    const std::string position,
    const std::string shape_function,
    const std::string dof_value,
    const std::string JxW,
    const std::string value,
    const std::string gradient,
    const std::string symmetric_gradient,
    const std::string divergence,
    const std::string curl,
    const std::string hessian,
    const std::string laplacian,
    const std::string third_derivative,
    const std::string normal,
    const std::string volume,
    const std::string boundary,
    const std::string interface,
    const std::string infinitesimal_element_volume,
    const std::string infinitesimal_element_boundary_area,
    const std::string infinitesimal_element_interface_area)
    : solution_field(solution_field)
    , test_function(test_function)
    , trial_solution(trial_solution)
    , position(position)
    , shape_function(shape_function)
    , dof_value(dof_value)
    , JxW(JxW)
    , value(value)
    , gradient(gradient)
    , symmetric_gradient(symmetric_gradient)
    , divergence(divergence)
    , curl(curl)
    , hessian(hessian)
    , laplacian(laplacian)
    , third_derivative(third_derivative)
    , normal(normal)
    , volume(volume)
    , boundary(boundary)
    , interface(interface)
    , infinitesimal_element_volume(infinitesimal_element_volume)
    , infinitesimal_element_boundary_area(infinitesimal_element_boundary_area)
    , infinitesimal_element_interface_area(infinitesimal_element_interface_area)
  {}



  SymbolicNamesAscii::SymbolicNamesAscii(
    const std::string solution_field,
    const std::string test_function,
    const std::string trial_solution,
    const std::string position,
    const std::string shape_function,
    const std::string dof_value,
    const std::string JxW,
    const std::string value,
    const std::string gradient,
    const std::string symmetric_gradient,
    const std::string divergence,
    const std::string curl,
    const std::string hessian,
    const std::string laplacian,
    const std::string third_derivative,
    const std::string normal,
    const std::string volume,
    const std::string boundary,
    const std::string interface,
    const std::string infinitesimal_element_volume,
    const std::string infinitesimal_element_boundary_area,
    const std::string infinitesimal_element_interface_area)
    : SymbolicNames(solution_field,
                    test_function,
                    trial_solution,
                    position,
                    shape_function,
                    dof_value,
                    JxW,
                    value,
                    gradient,
                    symmetric_gradient,
                    divergence,
                    curl,
                    hessian,
                    laplacian,
                    third_derivative,
                    normal,
                    volume,
                    boundary,
                    interface,
                    infinitesimal_element_volume,
                    infinitesimal_element_boundary_area,
                    infinitesimal_element_interface_area)
  {}



  SymbolicNamesLaTeX::SymbolicNamesLaTeX(
    const std::string solution_field,
    const std::string test_function,
    const std::string trial_solution,
    const std::string position,
    const std::string shape_function,
    const std::string dof_value,
    const std::string JxW,
    const std::string value,
    const std::string gradient,
    const std::string symmetric_gradient,
    const std::string divergence,
    const std::string curl,
    const std::string hessian,
    const std::string laplacian,
    const std::string third_derivative,
    const std::string normal,
    const std::string volume,
    const std::string boundary,
    const std::string interface,
    const std::string infinitesimal_element_volume,
    const std::string infinitesimal_element_boundary_area,
    const std::string infinitesimal_element_interface_area)
    : SymbolicNames(solution_field,
                    test_function,
                    trial_solution,
                    position,
                    shape_function,
                    dof_value,
                    JxW,
                    value,
                    gradient,
                    symmetric_gradient,
                    divergence,
                    curl,
                    hessian,
                    laplacian,
                    third_derivative,
                    normal,
                    volume,
                    boundary,
                    interface,
                    infinitesimal_element_volume,
                    infinitesimal_element_boundary_area,
                    infinitesimal_element_interface_area)
  {}



  SymbolicDecorations::SymbolicDecorations(
    const SymbolicNamesAscii &naming_ascii,
    const SymbolicNamesLaTeX &naming_latex)
    : naming_ascii(naming_ascii)
    , naming_latex(naming_latex)
  {}

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE