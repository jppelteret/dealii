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

#ifndef dealii_weakforms_operators_h
#define dealii_weakforms_operators_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    enum class UnaryOpCodes
    {
      value,
      negate
    };


    enum class BinaryOpCodes
    {
      add,
      subtract,
      multiply
    };


    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    DeclExceptionMsg(
      ExcRequiresUnaryOperatorSpecialization,
      "This function is called in a class that is expected to be specialized "
      "for unary operations. All unary operators should be specialized, with "
      "a structure matching that of the exemplar class.");

    /**
     * Exception denoting that a unary operation has not been defined.
     */
    DeclException1(ExcUnaryOperatorNotDefined,
                   enum UnaryOpCodes,
                   << "The unary operator with code " +
                          Utilities::to_string(static_cast<int>(arg1)) +
                          " has not been defined.");

    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    DeclExceptionMsg(
      ExcRequiresBinaryOperatorSpecialization,
      "This function is called in a class that is expected to be specialized "
      "for binary operations. All binary operators should be specialized, with "
      "a structure matching that of the exemplar class.");

    /**
     * Exception denoting that a binary operation has not been defined.
     */
    DeclException1(ExcBinaryOperatorNotDefined,
                   enum BinaryOpCodes,
                   << "The binary operator with code " +
                          Utilities::to_string(static_cast<int>(arg1)) +
                          " has not been defined.");


    template <typename Op>
    class NoOp
    {};


    template <typename Op, enum UnaryOpCodes OpCode>
    class UnaryOp {
      public: UnaryOp(const Op &operand): operand(operand){
        AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());}

    std::string
    as_ascii() const
    {
      AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
      return "";
    }

    std::string
    as_latex() const
    {
      AssertThrow(false, ExcRequiresUnaryOperatorSpecialization());
      return "";
    }

  private:
    const Op &                     operand;
    static const enum UnaryOpCodes op_code = OpCode;
  }; // namespace Operators


  template <typename LhsOp, typename RhsOp, enum BinaryOpCodes OpCode>
  class BinaryOp {
    public: BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand):
      lhs_operand(lhs_operand),
    rhs_operand(rhs_operand){
      AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());}

  std::string
  as_ascii() const
  {
    AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
    return "";
  }

  std::string
  as_latex() const
  {
    AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
    return "";
  }

private:
  const LhsOp &                   lhs_operand;
  const RhsOp &                   rhs_operand;
  static const enum BinaryOpCodes op_code = OpCode;
}; // namespace WeakForms


template <typename... Args>
class Composition
{};

} // namespace Operators
} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_operators_h
