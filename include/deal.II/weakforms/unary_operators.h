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

#ifndef dealii_weakforms_unary_operators_h
#define dealii_weakforms_unary_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/spaces.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      template <typename Operand>
      std::string
      unary_op_operand_as_ascii(const Operand &operand)
      {
        return operand.get_symbol_ascii() + "{" + operand.get_field_ascii() +
               "}";
      }

      template <typename Operand>
      std::string
      unary_op_operand_as_latex(const Operand &operand)
      {
        return operand.get_symbol_latex() + "_{" + operand.get_field_latex() +
               "}";
      }


      /**
       *
       *
       * @param op A string that symbolises the operator that acts on the @p operand.
       * @param operand
       * @return std::string
       */
      std::string
      decorate_with_operator_ascii(const std::string &op,
                                   const std::string &operand)
      {
        if (op == "")
          return operand;

        return op + "(" + operand + ")";
      }


      /**
       *
       *
       * @param op A string that symbolises the operator that acts on the @p operand.
       * @param operand
       * @return std::string
       */
      std::string
      decorate_with_operator_latex(const std::string &op,
                                   const std::string &operand)
      {
        if (op == "")
          return operand;

        return op + "\\left\\(" + operand + "\\right\\)";
      }
    } // namespace internal

    /**
     * Extract the shape function values from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<Space<dim, spacedim>, UnaryOpCodes::value>
    {
      using Op = Space<dim, spacedim>;

    public:
      template <typename NumberType>
      using value_type = typename Op::template value_type<NumberType>;
      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      std::string
      as_ascii() const
      {
        const auto &naming = operand.get_naming_ascii();
        return internal::decorate_with_operator_ascii(
          naming.value, internal::unary_op_operand_as_ascii(operand));
      }

      std::string
      as_latex() const
      {
        const auto &naming = operand.get_naming_latex();
        return internal::decorate_with_operator_ascii(
          naming.value, internal::unary_op_operand_as_latex(operand));
      }

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell, 
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points, 
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_value(dof_index, q_point);
      }

      /**
       * Return all shape function values at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points, 
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        // TODO: Replace with range based loop
        for (unsigned int dof_index = 0; dof_index < fe_values.dofs_per_cell;
             ++dof_index)
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

        return out;
      }

    private:
      const Op &                     operand;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;
    };



    /**
     * Extract the shape function gradients from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<Space<dim, spacedim>, UnaryOpCodes::gradient>
    {
      using Op = Space<dim, spacedim>;

    public:
      template <typename NumberType>
      using value_type = typename Op::template gradient_type<NumberType>;
      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      std::string
      as_ascii() const
      {
        const auto &naming = operand.get_naming_ascii();
        return internal::decorate_with_operator_ascii(
          naming.gradient, internal::unary_op_operand_as_ascii(operand));
      }

      std::string
      as_latex() const
      {
        const auto &naming = operand.get_naming_latex();
        return internal::decorate_with_operator_ascii(
          naming.gradient, internal::unary_op_operand_as_latex(operand));
      }

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell, 
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points, 
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_grad(dof_index, q_point);
      }


      /**
       * Return all shape function gradients at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points, 
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.dofs_per_cell);

        // TODO: Replace with range based loop
        for (unsigned int dof_index = 0; dof_index < fe_values.dofs_per_cell;
             ++dof_index)
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

        return out;
      }

    private:
      const Op &                     operand;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;
    };

  } // namespace Operators


  /* ========== Values ========== */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::Space<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Space<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::Space<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Space<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::Space<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    AssertThrow(
      false,
      ExcMessage(
        "Need to define a special UnaryOp to get the solution value."));
    using Op     = Space<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }


  /* ========== Gradients ========== */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::Space<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Space<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::Space<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = Space<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::Space<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    AssertThrow(
      false,
      ExcMessage(
        "Need to define a special UnaryOp to get the solution gradient."));
    using Op     = Space<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_operators_h
