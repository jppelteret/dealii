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

#include <deal.II/weakforms/operators.h>
#include <deal.II/weakforms/spaces.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    template <int dim, int spacedim>
    class BinaryOp<Space<dim, spacedim>,
                   Space<dim, spacedim>,
                   BinaryOpCodes::add>
    {
      using LhsOp = Space<dim, spacedim>;
      using RhsOp = Space<dim, spacedim>;

    public:
      BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " + " + rhs_operand.as_ascii() + "]";
      }

      std::string
      as_latex() const
      {
        return "\\left\\[" +lhs_operand.as_latex() + " + " + rhs_operand.as_latex() + "\\right\\]";
      }

    private:
      const LhsOp &                   lhs_operand;
      const RhsOp &                   rhs_operand;
      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;
    };

  } // namespace Operators

} // namespace WeakForms


/* ===================== Define operator overloads ===================== */


template <int dim, int spacedim>
WeakForms::Operators::BinaryOp<WeakForms::Space<dim, spacedim>,
                               WeakForms::Space<dim, spacedim>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::TrialSolution<dim, spacedim> &lhs_op,
          const WeakForms::FieldSolution<dim, spacedim> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;
  using OpType =
    BinaryOp<Space<dim, spacedim>, Space<dim, spacedim>, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


template <int dim, int spacedim>
WeakForms::Operators::BinaryOp<WeakForms::Space<dim, spacedim>,
                               WeakForms::Space<dim, spacedim>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::FieldSolution<dim, spacedim> &lhs_op,
          const WeakForms::TrialSolution<dim, spacedim> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  // Use the other definition, keeping the trial solution on the LHS
  return rhs_op + lhs_op;
}


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
