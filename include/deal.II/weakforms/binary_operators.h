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
    template <int dim, int spacedim>
    class BinaryOp<Space<dim, spacedim>,
                   Space<dim, spacedim>,
                   BinaryOpCodes::add>
    {
      using LhsOp = Space<dim, spacedim>;
      using RhsOp = Space<dim, spacedim>;

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


/**
 * @brief Unary op + unary op
 *
 * @tparam LhsOp
 * @tparam LhsOpCode
 * @tparam RhsOp
 * @tparam RhsOpCode
 * @param lhs_op
 * @param rhs_op
 * @return WeakForms::Operators::BinaryOp<WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
 * WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
 * WeakForms::Operators::BinaryOpCodes::add>
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
                               WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &lhs_op,
          const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op + binary op
 *
 * @tparam LhsOp
 * @tparam LhsOpCode
 * @tparam RhsOp
 * @tparam RhsOpCode
 * @param lhs_op
 * @param rhs_op
 * @return WeakForms::Operators::BinaryOp<WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
 * WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
 * WeakForms::Operators::BinaryOpCodes::add>
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::add>
operator+(
  const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &          lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


// ~~~~~~~~~~~~~~~~
// TODO: Testing only! Remove this. Its absolute nonesense.

template <int dim, int spacedim>
WeakForms::Operators::BinaryOp<WeakForms::Space<dim, spacedim>,
                               WeakForms::Space<dim, spacedim>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::TrialSolution<dim, spacedim> &lhs_op,
          const WeakForms::FieldSolution<dim, spacedim> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOp  = Space<dim, spacedim>;
  using RhsOp  = Space<dim, spacedim>;
  using OpType = BinaryOp<LhsOp, RhsOp, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


// TODO: Testing only! Remove this. Its absolute nonesense.
template <int dim, int spacedim>
WeakForms::Operators::BinaryOp<WeakForms::Space<dim, spacedim>,
                               WeakForms::Space<dim, spacedim>,
                               WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::FieldSolution<dim, spacedim> &lhs_op,
          const WeakForms::TrialSolution<dim, spacedim> &rhs_op)
{
  // Use the other definition, keeping the trial solution on the LHS
  return rhs_op + lhs_op;
}


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
