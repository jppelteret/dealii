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
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>
#include <deal.II/weak_forms/utilities.h>

#include <type_traits>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    enum class BinaryOpCodes
    {
      /**
       * Add two operands together.
       */
      add,
      /**
       * Subtract one operand from another.
       */
      subtract,
      /**
       * Multiply two operands together.
       */
      multiply,
      /**
       * Multiply the operand by a constant factor.
       */
      scale
    };



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
                   << "The binary operator with code " << static_cast<int>(arg1)
                   << " has not been defined.");



    /**
     * @tparam Op
     * @tparam OpCode
     * @tparam UnderlyingType Underlying number type (double, std::complex<double>, etc.).
     * This is necessary because some specializations of the class do not use
     * the number type in the specialization itself, but they may rely on the
     * type in their definitions (e.g. class members).
     */
    template <typename LhsOp,
              typename RhsOp,
              enum BinaryOpCodes OpCode,
              typename UnderlyingType = void>
    class BinaryOp
    {
    public:
      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
      }

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
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    }; // class BinaryOp



    namespace internal
    {
      // Assume that everything is compatible to add together or subtract apart
      template <typename LhsOp, typename RhsOp, typename T = void>
      struct has_incompatible_spaces_for_addition_subtraction : std::false_type
      {};


      // Cannot add or subtract a test function and field solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_test_function<LhsOp>::value &&
                                is_field_solution<RhsOp>::value>::type>
        : std::true_type
      {};


      // Cannot add or subtract a test function and trial solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_test_function<LhsOp>::value &&
                                is_trial_solution<RhsOp>::value>::type>
        : std::true_type
      {};


      // Cannot add or subtract a field solution and trial solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_field_solution<LhsOp>::value &&
                                is_trial_solution<RhsOp>::value>::type>
        : std::true_type
      {};


      // Check a + (b1+b2)
      template <typename LhsOp, typename RhsOp1, typename RhsOp2>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        BinaryOp<RhsOp1, RhsOp2, BinaryOpCodes::add>,
        typename std::enable_if<
          has_incompatible_spaces_for_addition_subtraction<LhsOp,
                                                           RhsOp1>::value ||
          has_incompatible_spaces_for_addition_subtraction<LhsOp, RhsOp2>::
            value>::type> : std::true_type
      {};


      // Check a + (b1-b2)
      template <typename LhsOp, typename RhsOp1, typename RhsOp2>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        BinaryOp<RhsOp1, RhsOp2, BinaryOpCodes::subtract>,
        typename std::enable_if<
          has_incompatible_spaces_for_addition_subtraction<LhsOp,
                                                           RhsOp1>::value ||
          has_incompatible_spaces_for_addition_subtraction<LhsOp, RhsOp2>::
            value>::type> : std::true_type
      {};


      // Deal with the combinatorics of the above by checking both combinations
      // [Lhs,Rhs] and [Rhs,Lhs] together. We negate the condition at the same
      // time.
      template <typename LhsOp, typename RhsOp>
      struct has_compatible_spaces_for_addition_subtraction
        : std::conditional<
            has_incompatible_spaces_for_addition_subtraction<LhsOp,
                                                             RhsOp>::value ||
              has_incompatible_spaces_for_addition_subtraction<RhsOp,
                                                               LhsOp>::value,
            std::false_type,
            std::true_type>::type
      {};

    } // namespace internal


    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp, RhsOp, BinaryOpCodes::add>
    {
      static_assert(
        internal::has_compatible_spaces_for_addition_subtraction<LhsOp,
                                                                 RhsOp>::value,
        "It is not permissible to add incompatible spaces together.");

    public:
      template <typename NumberType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<NumberType>>() +
        std::declval<typename RhsOp::template value_type<NumberType>>());

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank =
        WeakForms::Utilities::IndexContraction<LhsOp, RhsOp>::result_rank;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        // Assert(&lhs_operand.get_decorator() == &rhs_operand.get_decorator(),
        // ExcMessage("LHS and RHS operands do not use the same decorator."));
        return lhs_operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " + " + rhs_operand.as_ascii() +
               "]";
      }

      std::string
      as_latex() const
      {
        const std::string lbrace = Utilities::LaTeX::l_square_brace;
        const std::string rbrace = Utilities::LaTeX::r_square_brace;

        return lbrace + lhs_operand.as_latex() + " + " +
               rhs_operand.as_latex() + rbrace;
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return lhs_operand.get_update_flags() | rhs_operand.get_update_flags();
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
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };


    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp, RhsOp, BinaryOpCodes::subtract>
    {
      static_assert(
        internal::has_compatible_spaces_for_addition_subtraction<LhsOp,
                                                                 RhsOp>::value,
        "It is not permissible to subtract incompatible spaces from one another.");

    public:
      template <typename NumberType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<NumberType>>() -
        std::declval<typename RhsOp::template value_type<NumberType>>());

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank =
        WeakForms::Utilities::IndexContraction<LhsOp, RhsOp>::result_rank;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::subtract;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        // Assert(&lhs_operand.get_decorator() == &rhs_operand.get_decorator(),
        // ExcMessage("LHS and RHS operands do not use the same decorator."));
        return lhs_operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " - " + rhs_operand.as_ascii() +
               "]";
      }

      std::string
      as_latex() const
      {
        const std::string lbrace = Utilities::LaTeX::l_square_brace;
        const std::string rbrace = Utilities::LaTeX::r_square_brace;

        return lbrace + lhs_operand.as_latex() + " - " +
               rhs_operand.as_latex() + rbrace;
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return lhs_operand.get_update_flags() | rhs_operand.get_update_flags();
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
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };


    template <typename LhsOp, typename RhsOp>
    class BinaryOp<LhsOp, RhsOp, BinaryOpCodes::multiply, void>
    {
    public:
      template <typename NumberType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<NumberType>>() *
        std::declval<typename RhsOp::template value_type<NumberType>>());

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank =
        WeakForms::Utilities::IndexContraction<LhsOp, RhsOp>::result_rank;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::multiply;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        // Assert(&lhs_operand.get_decorator() == &rhs_operand.get_decorator(),
        // ExcMessage("LHS and RHS operands do not use the same decorator."));
        return lhs_operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        return "[" + lhs_operand.as_ascii() + " * " + rhs_operand.as_ascii() +
               "]";
      }

      std::string
      as_latex() const
      {
        const SymbolicDecorations &decorator = get_decorator();

        const std::string lbrace = Utilities::LaTeX::l_square_brace;
        const std::string rbrace = Utilities::LaTeX::r_square_brace;

        constexpr unsigned int n_contracting_indices =
          WeakForms::Utilities::IndexContraction<LhsOp,
                                                 RhsOp>::n_contracting_indices;
        const std::string symb_mult =
          Utilities::LaTeX::get_symbol_multiply(n_contracting_indices);
        return lbrace + lhs_operand.as_latex() + symb_mult +
               rhs_operand.as_latex() + rbrace;
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return lhs_operand.get_update_flags() | rhs_operand.get_update_flags();
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
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };

  } // namespace Operators

} // namespace WeakForms



/* ===================== Define operator overloads ===================== */
// See https://stackoverflow.com/a/12782697 for using multiple parameter packs


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
          typename LhsUnderlyingType,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode,
          typename RhsUnderlyingType>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode, LhsUnderlyingType>,
  WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode, RhsUnderlyingType>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(
  const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode, LhsUnderlyingType>
    &lhs_op,
  const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode, RhsUnderlyingType>
    &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode, LhsUnderlyingType>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode, RhsUnderlyingType>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op * binary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::UnaryOpCodes LhsOpCode,
          typename LhsUnderlyingType,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode,
          typename RhsUnderlyingType>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode, LhsUnderlyingType>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(
  const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode, LhsUnderlyingType>
    &lhs_op,
  const WeakForms::Operators::
    BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = UnaryOp<LhsOp, LhsOpCode, LhsUnderlyingType>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op * unary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename LhsUnderlyingType,
          typename RhsOp,
          enum WeakForms::Operators::UnaryOpCodes RhsOpCode,
          typename RhsUnderlyingType>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType>,
  WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode, RhsUnderlyingType>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(
  const WeakForms::Operators::
    BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType> &lhs_op,
  const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode, RhsUnderlyingType>
    &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType>;
  using RhsOpType = UnaryOp<RhsOp, RhsOpCode, RhsUnderlyingType>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}



/**
 * @brief Binary op * binary op
 */
template <typename LhsOp1,
          typename LhsOp2,
          enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
          typename LhsUnderlyingType,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode,
          typename RhsUnderlyingType>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(const WeakForms::Operators::
            BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType> &lhs_op,
          const WeakForms::Operators::
            BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType>;
  using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}



// #ifndef DOXYGEN


// namespace WeakForms
// {
//   template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
//   struct is_test_function<
//     Operators::BinaryOp<TestFunction<dim, spacedim>, OpCode>> :
//     std::true_type
//   {};

//   template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
//   struct is_trial_solution<
//     Operators::BinaryOp<TrialSolution<dim, spacedim>, OpCode>> :
//     std::true_type
//   {};

//   template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
//   struct is_field_solution<
//     Operators::BinaryOp<FieldSolution<dim, spacedim>, OpCode>> :
//     std::true_type
//   {};

// } // namespace WeakForms


// #endif // DOXYGEN



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename... Args>
  struct is_binary_op<Operators::BinaryOp<Args...>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
