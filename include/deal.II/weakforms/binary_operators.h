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
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp, RhsOp, BinaryOpCodes::add>
    {
      // using LhsOp = Space<dim, spacedim>;
      // using RhsOp = Space<dim, spacedim>;

      // using value_type = decltype(std::declval<typename LhsOp::value_type>()
      // +
      //                             std::declval<typename
      //                             RhsOp::value_type>());

    public:
      template <typename NumberType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<NumberType>>() +
        std::declval<typename RhsOp::template value_type<NumberType>>());

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;

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

      template <typename NumberType>
      value_type<NumberType>
      operator()(
        const typename LhsOp::template value_type<NumberType> &lhs_value,
        const typename RhsOp::template value_type<NumberType> &rhs_value)
      {
        return lhs_value + rhs_value;
      }

      template <typename NumberType>
      return_type<NumberType>
      operator()(
        const typename LhsOp::template return_type<NumberType> &lhs_value,
        const typename RhsOp::template return_type<NumberType> &rhs_value)
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<NumberType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(this->operator()(lhs_value[i], rhs_value[i]));

        return out;
      }

    private:
      const LhsOp &lhs_operand;
      const RhsOp &rhs_operand;
    };


    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp, RhsOp, BinaryOpCodes::subtract>
    {
      // using LhsOp = Space<dim, spacedim>;
      // using RhsOp = Space<dim, spacedim>;

    public:
      template <typename NumberType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<NumberType>>() -
        std::declval<typename RhsOp::template value_type<NumberType>>());

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::subtract;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " - " + rhs_operand.as_ascii() +
               "]";
      }

      std::string
      as_latex() const
      {
        return "\\left\\[" + lhs_operand.as_latex() + " - " +
               rhs_operand.as_latex() + "\\right\\]";
      }

      template <typename NumberType>
      value_type<NumberType>
      operator()(
        const typename LhsOp::template value_type<NumberType> &lhs_value,
        const typename RhsOp::template value_type<NumberType> &rhs_value)
      {
        return lhs_value - rhs_value;
      }

      template <typename NumberType>
      return_type<NumberType>
      operator()(
        const typename LhsOp::template return_type<NumberType> &lhs_value,
        const typename RhsOp::template return_type<NumberType> &rhs_value)
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<NumberType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(this->operator()(lhs_value[i], rhs_value[i]));

        return out;
      }

    private:
      const LhsOp &lhs_operand;
      const RhsOp &rhs_operand;
    };


    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp, RhsOp, BinaryOpCodes::multiply>
    {
      // using LhsOp = Space<dim, spacedim>;
      // using RhsOp = Space<dim, spacedim>;

    public:
      template <typename NumberType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<NumberType>>() *
        std::declval<typename RhsOp::template value_type<NumberType>>());

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::multiply;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " * " + rhs_operand.as_ascii() +
               "]";
      }

      std::string
      as_latex() const
      {
        return "\\left\\[" + lhs_operand.as_latex() + " * " +
               rhs_operand.as_latex() + "\\right\\]";
      }

      template <typename NumberType>
      value_type<NumberType>
      operator()(
        const typename LhsOp::template value_type<NumberType> &lhs_value,
        const typename RhsOp::template value_type<NumberType> &rhs_value)
      {
        return lhs_value * rhs_value;
      }

      template <typename NumberType>
      return_type<NumberType>
      operator()(
        const typename LhsOp::template return_type<NumberType> &lhs_value,
        const typename RhsOp::template return_type<NumberType> &rhs_value)
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<NumberType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(this->operator()(lhs_value[i], rhs_value[i]));

        return out;
      }

    private:
      const LhsOp &lhs_operand;
      const RhsOp &rhs_operand;
    };

  } // namespace Operators

} // namespace WeakForms


/* ===================== Define operator overloads ===================== */


/* ---------------------------- Addition ---------------------------- */


/**
 * @brief Unary op + unary op
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



/**
 * @brief Binary op + unary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::add>
operator+(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &          rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op + binary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::add>
operator+(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


/* ---------------------------- Subtraction ---------------------------- */


/**
 * @brief Unary op - unary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
                               WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
                               WeakForms::Operators::BinaryOpCodes::subtract>
operator-(const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &lhs_op,
          const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op - binary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::subtract>
operator-(
  const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &          lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op - unary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::subtract>
operator-(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &          rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op - binary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::subtract>
operator-(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

  return OpType(lhs_op, rhs_op);
}


/* ---------------------------- Multiplication ---------------------------- */


/**
 * @brief Unary op * unary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
                               WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
                               WeakForms::Operators::BinaryOpCodes::multiply>
operator*(const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &lhs_op,
          const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op * binary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(
  const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &          lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op * unary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &          rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op * binary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
