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

#ifndef dealii_weakforms_binary_operators_h
#define dealii_weakforms_binary_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

// TODO: Move FeValuesViews::[Scalar/Vector/...]::Output<> into another header??
#include <deal.II/fe/fe_values.h>

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/spaces.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    template <int dim, int spacedim, typename T1, typename T2>
    class BinaryOp<Space<dim, spacedim, T1>,
                   Space<dim, spacedim, T2>,
                   BinaryOpCodes::add>
    {
      using LhsOp = Space<dim, spacedim, T1>;
      using RhsOp = Space<dim, spacedim, T2>;

      // using value_type = decltype(std::declval<typename LhsOp::value_type>()
      // +
      //                             std::declval<typename
      //                             RhsOp::value_type>());

    public:
      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " + " + rhs_operand.as_ascii() +
               "]";
      }

      std::string
      as_latex() const
      {
        return "\\left\\[" + lhs_operand.as_latex() + " + " +
               rhs_operand.as_latex() + "\\right\\]";
      }

      // value_type
      // operator()(const typename LhsOp::value_type &lhs_value,
      //            const typename RhsOp::value_type &rhs_value)
      // {
      //   return lhs_value + rhs_value;
      // }

    private:
      const LhsOp &                   lhs_operand;
      const RhsOp &                   rhs_operand;
      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;
    };

  } // namespace Operators

} // namespace WeakForms


/* ===================== Define operator overloads ===================== */


// TODO: Testing only! Remove this. Its absolute nonesense.
template <int dim, int spacedim, typename T1, typename T2>
WeakForms::Operators::BinaryOp<WeakForms::Space<dim, spacedim, T1>,
                               WeakForms::Space<dim, spacedim, T2>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::TrialSolution<dim, spacedim, T1> &lhs_op,
          const WeakForms::FieldSolution<dim, spacedim, T2> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOp  = Space<dim, spacedim, T1>;
  using RhsOp  = Space<dim, spacedim, T2>;
  using OpType = BinaryOp<LhsOp, RhsOp, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


// TODO: Testing only! Remove this. Its absolute nonesense.
template <int dim, int spacedim, typename T1, typename T2>
WeakForms::Operators::BinaryOp<WeakForms::Space<dim, spacedim, T1>,
                               WeakForms::Space<dim, spacedim, T2>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::FieldSolution<dim, spacedim, T1> &lhs_op,
          const WeakForms::TrialSolution<dim, spacedim, T2> &rhs_op)
{
  // Use the other definition, keeping the trial solution on the LHS
  return rhs_op + lhs_op;
}


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
