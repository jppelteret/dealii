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

#include <deal.II/meshworker/scratch_data.h>

#include <boost/core/demangle.hpp> // DEBUGGING

#include <deal.II/weak_forms/solution_storage.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/symbolic_operators.h>
#include <deal.II/weak_forms/type_traits.h>
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
      /**
       * Cross product (between two vector operands)
       */
      // cross_product
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
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    template <typename LhsOpType, typename RhsOpType>
    DeclException2(
      ExcRequiresBinaryOperatorSpecialization2,
      LhsOpType,
      RhsOpType,
      << "This function is called in a class that is expected to be specialized "
      << "for binary operations. All binary operators should be specialized, with "
      << "a structure matching that of the exemplar class.\n\n"
      << "LHS op type" << boost::core::demangle(typeid(arg1).name()) << "\n\n"
      << "RHS op type" << boost::core::demangle(typeid(arg2).name()) << "\n");


    /**
     * Exception denoting that a binary operation has not been defined.
     */
    DeclException1(ExcBinaryOperatorNotDefined,
                   enum BinaryOpCodes,
                   << "The binary operator with code " << static_cast<int>(arg1)
                   << " has not been defined.");



    namespace internal
    {
      /**
       * Helper to return values at all quadrature points
       */
      template <typename LhsOpType, typename RhsOpType, typename T = void>
      struct BinaryOpHelper;


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: Neither operand is a field solution
       */
      template <typename LhsOpType, typename RhsOpType>
      struct BinaryOpHelper<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOpType>::value &&
          !is_test_function_or_trial_solution_op<RhsOpType>::value &&
          !evaluates_with_scratch_data<LhsOpType>::value &&
          !evaluates_with_scratch_data<RhsOpType>::value>::type>
      {
        template <typename ScalarType,
                  typename BinaryOpType,
                  int dim,
                  int spacedim>
        static typename BinaryOpType::template return_type<ScalarType>
        apply(const BinaryOpType &               op,
              const LhsOpType &                  lhs_operand,
              const RhsOpType &                  rhs_operand,
              const FEValuesBase<dim, spacedim> &fe_values)
        {
          return op.template operator()<ScalarType>(
            lhs_operand.template operator()<ScalarType>(fe_values),
            rhs_operand.template operator()<ScalarType>(fe_values));
        }

        template <typename ScalarType,
                  typename BinaryOpType,
                  int dim,
                  int spacedim>
        static typename BinaryOpType::template return_type<ScalarType>
        apply(const BinaryOpType &                    op,
              const LhsOpType &                       lhs_operand,
              const RhsOpType &                       rhs_operand,
              const FEValuesBase<dim, spacedim> &     fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)scratch_data;
          (void)solution_names;

          return apply<ScalarType>(op, lhs_operand, rhs_operand, fe_values);
        }
      };


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: LHS operand is a field solution
       */
      template <typename LhsOpType, typename RhsOpType>
      struct BinaryOpHelper<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOpType>::value &&
          !is_test_function_or_trial_solution_op<RhsOpType>::value &&
          evaluates_with_scratch_data<LhsOpType>::value &&
          !evaluates_with_scratch_data<RhsOpType>::value>::type>
      {
        template <typename ScalarType,
                  typename BinaryOpType,
                  int dim,
                  int spacedim>
        static typename BinaryOpType::template return_type<ScalarType>
        apply(const BinaryOpType &                    op,
              const LhsOpType &                       lhs_operand,
              const RhsOpType &                       rhs_operand,
              const FEValuesBase<dim, spacedim> &     fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          return op.template operator()<ScalarType>(
            lhs_operand.template operator()<ScalarType>(scratch_data,
                                                        solution_names),
            rhs_operand.template operator()<ScalarType>(fe_values));
        }
      };


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: RHS operand is a field solution
       */
      template <typename LhsOpType, typename RhsOpType>
      struct BinaryOpHelper<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOpType>::value &&
          !is_test_function_or_trial_solution_op<RhsOpType>::value &&
          !evaluates_with_scratch_data<LhsOpType>::value &&
          evaluates_with_scratch_data<RhsOpType>::value>::type>
      {
        template <typename ScalarType,
                  typename BinaryOpType,
                  int dim,
                  int spacedim>
        static typename BinaryOpType::template return_type<ScalarType>
        apply(const BinaryOpType &                    op,
              const LhsOpType &                       lhs_operand,
              const RhsOpType &                       rhs_operand,
              const FEValuesBase<dim, spacedim> &     fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          return op.template operator()<ScalarType>(
            lhs_operand.template operator()<ScalarType>(fe_values),
            rhs_operand.template operator()<ScalarType>(scratch_data,
                                                        solution_names));
        }
      };


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: Both operands are field solutions
       */
      template <typename LhsOpType, typename RhsOpType>
      struct BinaryOpHelper<
        LhsOpType,
        RhsOpType,
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOpType>::value &&
          !is_test_function_or_trial_solution_op<RhsOpType>::value &&
          evaluates_with_scratch_data<LhsOpType>::value &&
          evaluates_with_scratch_data<RhsOpType>::value>::type>
      {
        template <typename ScalarType,
                  typename BinaryOpType,
                  int dim,
                  int spacedim>
        static typename BinaryOpType::template return_type<ScalarType>
        apply(const BinaryOpType &                    op,
              const LhsOpType &                       lhs_operand,
              const RhsOpType &                       rhs_operand,
              const FEValuesBase<dim, spacedim> &     fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)fe_values;

          return op.template operator()<ScalarType>(
            lhs_operand.template operator()<ScalarType>(scratch_data,
                                                        solution_names),
            rhs_operand.template operator()<ScalarType>(scratch_data,
                                                        solution_names));
        }
      };
    } // namespace internal



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
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = OpCode;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {
        std::cout << "LHS op type: "
                  << boost::core::demangle(typeid(lhs_operand).name())
                  << std::endl;
        std::cout << "RHS op type: "
                  << boost::core::demangle(typeid(rhs_operand).name())
                  << std::endl;
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
        // AssertThrow(false,
        // ExcRequiresBinaryOperatorSpecialization2<LhsOp,RhsOp>(lhs_operand,
        // rhs_operand));
      }

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        (void)decorator;
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
        return "";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        (void)decorator;
        AssertThrow(false, ExcRequiresBinaryOperatorSpecialization());
        return "";
      }

      // const LhsOp &
      // get_lhs_operand () const
      // {
      //   return lhs_operand;
      // }

      // const RhsOp &
      // get_rhs_operand () const
      // {
      //   return rhs_operand;
      // }

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
        typename std::enable_if<is_test_function_op<LhsOp>::value &&
                                is_field_solution_op<RhsOp>::value>::type>
        : std::true_type
      {};


      // Cannot add or subtract a test function and trial solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_test_function_op<LhsOp>::value &&
                                is_trial_solution_op<RhsOp>::value>::type>
        : std::true_type
      {};


      // Cannot add or subtract a field solution and trial solution
      template <typename LhsOp, typename RhsOp>
      struct has_incompatible_spaces_for_addition_subtraction<
        LhsOp,
        RhsOp,
        typename std::enable_if<is_field_solution_op<LhsOp>::value &&
                                is_trial_solution_op<RhsOp>::value>::type>
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


      template <typename T, typename U = void>
      struct has_addition_or_subtraction_op_code : std::false_type
      {};

      template <typename T>
      struct has_addition_or_subtraction_op_code<
        T,
        typename std::enable_if<T::op_code == Operators::BinaryOpCodes::add ||
                                T::op_code ==
                                  Operators::BinaryOpCodes::subtract>::type>
        : std::true_type
      {};


      template <typename LhsOp, typename RhsOp>
      struct is_valid_binary_integral_op
        : std::conditional<
            // Both operands are standard integrals,
            // i.e. the case   assembler += ().dV + ().dV
            (is_unary_integral_op<LhsOp>::value
               &&is_unary_integral_op<RhsOp>::value) ||
              // The LHS op is a composite integral operation and the second a
              // unary one, i.e. the case  assembler += (().dV + ().dV) + ().dV
              (is_binary_op<LhsOp>::value
                 &&is_unary_integral_op<RhsOp>::value) ||
              // The LHS op is a composite integral operation and the second a
              // unary one, i.e. the case  assembler += ().dV + (().dV + ().dV)
              (is_binary_op<RhsOp>::value &&is_unary_integral_op<LhsOp>::value),
            std::true_type,
            std::false_type>::type
      {};


      // template <typename LhsOp, typename RhsOp, typename T = void>
      // struct is_valid_binary_integral_op : std::false_type
      // {};


      // // Composition of functor types, as well as
      // // composition of integral types
      // template <typename LhsOp, typename RhsOp>
      // struct is_valid_binary_integral_op<
      //   LhsOp,
      //   RhsOp,
      //   typename std::enable_if<
      //     // Both operands are unary functors, e.g. A + B
      //     (is_unary_op<LhsOp>::value &&is_unary_op<RhsOp>::value) ||
      //     // The LHS op is a binary functor and the second
      //     // unary one, i.e. (A + B) + C
      //     (is_binary_op<LhsOp>::value &&is_unary_op<RhsOp>::value) ||
      //     // The RHS op is a binary functor and the first
      //     // unary one, i.e. A + (B + C)
      //     (is_binary_op<RhsOp>::value &&is_unary_op<LhsOp>::value)>::type>
      //   : std::true_type
      // {};


      // // Composition of integral types
      // template <typename LhsOp, typename RhsOp>
      // struct is_valid_binary_integral_op<
      //   LhsOp,
      //   RhsOp,
      //   typename std::enable_if<
      //     // Both operands are standard integrals,
      //     // i.e. the case   assembler += ().dV + ().dV
      //     (is_unary_integral_op<LhsOp>::value
      //        &&is_unary_integral_op<RhsOp>::value) ||
      //     // The LHS op is a composite integral operation and the second a
      //     // unary one, i.e. the case  assembler += (().dV + ().dV) + ().dV
      //     (is_binary_integral_op<LhsOp>::value
      //        &&is_unary_integral_op<RhsOp>::value) ||
      //     // The RHS op is a composite integral operation and the first a
      //     // unary one, i.e. the case  assembler += ().dV + (().dV + ().dV)
      //     (is_binary_integral_op<RhsOp>::value
      //        &&is_unary_integral_op<LhsOp>::value)>::type> : std::true_type
      // {};

    } // namespace internal

  } // namespace Operators
} // namespace WeakForms



/* ================== Specialization of binary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /**
     * Addition operator for symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<
      LhsOp,
      RhsOp,
      BinaryOpCodes::add,
      typename std::enable_if<
        internal::is_valid_binary_integral_op<LhsOp, RhsOp>::value>::type>
    {
    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_ascii(decorator) + " + " +
               rhs_operand.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_latex(decorator) + " + " +
               rhs_operand.as_latex(decorator);
      }

      // These need to be exposed for the assembler to accumulate
      // the compound integral expression.
      const LhsOp &
      get_lhs_operand() const
      {
        return lhs_operand;
      }

      const RhsOp &
      get_rhs_operand() const
      {
        return rhs_operand;
      }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Subtraction operator for symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<
      LhsOp,
      RhsOp,
      BinaryOpCodes::subtract,
      typename std::enable_if<
        internal::is_valid_binary_integral_op<LhsOp, RhsOp>::value>::type>
    {
    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::subtract;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_ascii(decorator) + " - " +
               rhs_operand.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return lhs_operand.as_latex(decorator) + " - " +
               rhs_operand.as_latex(decorator);
      }

      // These need to be exposed for the assembler to accumulate
      // the compound integral expression.
      const LhsOp &
      get_lhs_operand() const
      {
        return lhs_operand;
      }

      const RhsOp &
      get_rhs_operand() const
      {
        return rhs_operand;
      }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Multiplication operator for symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<
      LhsOp,
      RhsOp,
      BinaryOpCodes::multiply,
      typename std::enable_if<
        internal::is_valid_binary_integral_op<LhsOp, RhsOp>::value>::type>
    {
      static_assert(!is_unary_integral_op<LhsOp>::value &&
                      !is_unary_integral_op<RhsOp>::value,
                    "Multiplication of symbolic integrals is not permitted.");

    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::multiply;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Addition operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<
      LhsOp,
      RhsOp,
      BinaryOpCodes::add,
      typename std::enable_if<!is_unary_integral_op<LhsOp>::value &&
                              !is_unary_integral_op<RhsOp>::value>::type>
    {
      static_assert(
        internal::has_compatible_spaces_for_addition_subtraction<LhsOp,
                                                                 RhsOp>::value,
        "It is not permissible to add incompatible spaces together.");

    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::add;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() +
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      template <typename ScalarType>
      using return_type = std::vector<value_type<ScalarType>>;

      static const int rank =
        WeakForms::Utilities::IndexContraction<LhsOp, RhsOp>::result_rank;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " + " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const std::string lbrace = Utilities::LaTeX::l_square_brace;
        const std::string rbrace = Utilities::LaTeX::r_square_brace;

        return lbrace + lhs_operand.as_latex(decorator) + " + " +
               rhs_operand.as_latex(decorator) + rbrace;
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return lhs_operand.get_update_flags() | rhs_operand.get_update_flags();
      }


      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value + rhs_value;
      }

      template <typename ScalarType>
      return_type<ScalarType>
      operator()(
        const typename LhsOp::template return_type<ScalarType> &lhs_value,
        const typename RhsOp::template return_type<ScalarType> &rhs_value) const
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<ScalarType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(
            this->template operator()<ScalarType>(lhs_value[i], rhs_value[i]));

        return out;
      }


      /**
       * Return values at all quadrature points
       *
       * It is expected that this operator never be directly called on a
       * test function or trial solution, but rather that the latter be unpacked
       * manually within the assembler itself.
       * We also cannot expose this function when the operand types are
       * symbolic integrals.
       */
      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const ->
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOp>::value &&
            !is_test_function_or_trial_solution_op<RhsOp>::value &&
            !evaluates_with_scratch_data<LhsOp>::value &&
            !evaluates_with_scratch_data<RhsOp>::value,
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpHelper<LhsOp, RhsOp>::template apply<
          ScalarType>(*this, lhs_operand, rhs_operand, fe_values);
      }

      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &     fe_values,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOp>::value &&
            !is_test_function_or_trial_solution_op<RhsOp>::value &&
            (evaluates_with_scratch_data<LhsOp>::value ||
             evaluates_with_scratch_data<RhsOp>::value),
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpHelper<LhsOp, RhsOp>::template apply<
          ScalarType>(*this,
                      lhs_operand,
                      rhs_operand,
                      fe_values,
                      scratch_data,
                      solution_names);
      }

      // const LhsOp &
      // get_lhs_operand () const
      // {
      //   return lhs_operand;
      // }

      // const RhsOp &
      // get_rhs_operand () const
      // {
      //   return rhs_operand;
      // }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Subtraction operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<
      LhsOp,
      RhsOp,
      BinaryOpCodes::subtract,
      typename std::enable_if<!is_unary_integral_op<LhsOp>::value &&
                              !is_unary_integral_op<RhsOp>::value>::type>
    {
      static_assert(
        internal::has_compatible_spaces_for_addition_subtraction<LhsOp,
                                                                 RhsOp>::value,
        "It is not permissible to subtract incompatible spaces from one another.");

    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::subtract;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() -
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      template <typename ScalarType>
      using return_type = std::vector<value_type<ScalarType>>;

      static const int rank =
        WeakForms::Utilities::IndexContraction<LhsOp, RhsOp>::result_rank;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " - " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const std::string lbrace = Utilities::LaTeX::l_square_brace;
        const std::string rbrace = Utilities::LaTeX::r_square_brace;

        return lbrace + lhs_operand.as_latex(decorator) + " - " +
               rhs_operand.as_latex(decorator) + rbrace;
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return lhs_operand.get_update_flags() | rhs_operand.get_update_flags();
      }


      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value - rhs_value;
      }

      template <typename ScalarType>
      return_type<ScalarType>
      operator()(
        const typename LhsOp::template return_type<ScalarType> &lhs_value,
        const typename RhsOp::template return_type<ScalarType> &rhs_value) const
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<ScalarType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(
            this->template operator()<ScalarType>(lhs_value[i], rhs_value[i]));

        return out;
      }


      /**
       * Return values at all quadrature points
       *
       * It is expected that this operator never be directly called on a
       * test function or trial solution, but rather that the latter be unpacked
       * manually within the assembler itself.
       */
      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const ->
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOp>::value &&
            !is_test_function_or_trial_solution_op<RhsOp>::value &&
            !evaluates_with_scratch_data<LhsOp>::value &&
            !evaluates_with_scratch_data<RhsOp>::value,
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpHelper<LhsOp, RhsOp>::template apply<
          ScalarType>(*this, lhs_operand, rhs_operand, fe_values);
      }

      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &     fe_values,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOp>::value &&
            !is_test_function_or_trial_solution_op<RhsOp>::value &&
            (evaluates_with_scratch_data<LhsOp>::value ||
             evaluates_with_scratch_data<RhsOp>::value),
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpHelper<LhsOp, RhsOp>::template apply<
          ScalarType>(*this,
                      lhs_operand,
                      rhs_operand,
                      fe_values,
                      scratch_data,
                      solution_names);
      }

      // const LhsOp &
      // get_lhs_operand () const
      // {
      //   return lhs_operand;
      // }

      // const RhsOp &
      // get_rhs_operand () const
      // {
      //   return rhs_operand;
      // }

    private:
      const LhsOp lhs_operand;
      const RhsOp rhs_operand;
    };



    /**
     * Multiplication operator for integrands of symbolic integrals
     */
    template <typename LhsOp, typename RhsOp>
    class BinaryOp<
      LhsOp,
      RhsOp,
      BinaryOpCodes::multiply,
      typename std::enable_if<!is_unary_integral_op<LhsOp>::value &&
                              !is_unary_integral_op<RhsOp>::value>::type>
    {
    public:
      using LhsOpType = LhsOp;
      using RhsOpType = RhsOp;

      static const enum BinaryOpCodes op_code = BinaryOpCodes::multiply;

      template <typename ScalarType>
      using value_type = decltype(
        std::declval<typename LhsOp::template value_type<ScalarType>>() *
        std::declval<typename RhsOp::template value_type<ScalarType>>());

      template <typename ScalarType>
      using return_type = std::vector<value_type<ScalarType>>;

      static const int rank =
        WeakForms::Utilities::IndexContraction<LhsOp, RhsOp>::result_rank;

      explicit BinaryOp(const LhsOp &lhs_operand, const RhsOp &rhs_operand)
        : lhs_operand(lhs_operand)
        , rhs_operand(rhs_operand)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "[" + lhs_operand.as_ascii(decorator) + " * " +
               rhs_operand.as_ascii(decorator) + "]";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const std::string lbrace = Utilities::LaTeX::l_square_brace;
        const std::string rbrace = Utilities::LaTeX::r_square_brace;

        constexpr unsigned int n_contracting_indices =
          WeakForms::Utilities::IndexContraction<LhsOp,
                                                 RhsOp>::n_contracting_indices;
        const std::string symb_mult =
          Utilities::LaTeX::get_symbol_multiply(n_contracting_indices);
        return lbrace + lhs_operand.as_latex(decorator) + symb_mult +
               rhs_operand.as_latex(decorator) + rbrace;
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return lhs_operand.get_update_flags() | rhs_operand.get_update_flags();
      }

      template <typename ScalarType>
      value_type<ScalarType>
      operator()(
        const typename LhsOp::template value_type<ScalarType> &lhs_value,
        const typename RhsOp::template value_type<ScalarType> &rhs_value) const
      {
        return lhs_value * rhs_value;
      }

      template <typename ScalarType>
      return_type<ScalarType>
      operator()(
        const typename LhsOp::template return_type<ScalarType> &lhs_value,
        const typename RhsOp::template return_type<ScalarType> &rhs_value) const
      {
        Assert(lhs_value.size() == rhs_value.size(),
               ExcDimensionMismatch(lhs_value.size(), rhs_value.size()));

        return_type<ScalarType> out;
        const unsigned int      size = lhs_value.size();
        out.reserve(size);

        for (unsigned int i = 0; i < size; ++i)
          out.emplace_back(
            this->template operator()<ScalarType>(lhs_value[i], rhs_value[i]));

        return out;
      }


      /**
       * Return values at all quadrature points
       *
       * It is expected that this operator never be directly called on a
       * test function or trial solution, but rather that the latter be unpacked
       * manually within the assembler itself.
       */
      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const ->
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOp>::value &&
            !is_test_function_or_trial_solution_op<RhsOp>::value &&
            !evaluates_with_scratch_data<LhsOp>::value &&
            !evaluates_with_scratch_data<RhsOp>::value,
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpHelper<LhsOp, RhsOp>::template apply<
          ScalarType>(*this, lhs_operand, rhs_operand, fe_values);
      }

      template <typename ScalarType, int dim, int spacedim>
      auto
      operator()(const FEValuesBase<dim, spacedim> &     fe_values,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const ->
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<LhsOp>::value &&
            !is_test_function_or_trial_solution_op<RhsOp>::value &&
            (evaluates_with_scratch_data<LhsOp>::value ||
             evaluates_with_scratch_data<RhsOp>::value),
          return_type<ScalarType>>::type
      {
        return internal::BinaryOpHelper<LhsOp, RhsOp>::template apply<
          ScalarType>(*this,
                      lhs_operand,
                      rhs_operand,
                      fe_values,
                      scratch_data,
                      solution_names);
      }

      // const LhsOp &
      // get_lhs_operand () const
      // {
      //   return lhs_operand;
      // }

      // const RhsOp &
      // get_rhs_operand () const
      // {
      //   return rhs_operand;
      // }

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
          enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
          typename... LhsOpArgs,
          typename RhsOp,
          enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,
          typename... RhsOpArgs>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,
  WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,
  WeakForms::Operators::BinaryOpCodes::add>
operator+(const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>
            &lhs_op,
          const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>
            &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;
  using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op + binary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
          typename... LhsOpArgs,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::add>
operator+(
  const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>
    &                                                              lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;
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
          enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,
          typename... RhsOpArgs>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,
  WeakForms::Operators::BinaryOpCodes::add>
operator+(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>
    &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;
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


// /**
//  * @brief Unary op + unary op
//  */
// template <typename LhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
//           typename RhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<WeakForms::Operators::SymbolicOp<LhsOp,
// LhsOpCode>,
//                                WeakForms::Operators::SymbolicOp<RhsOp,
//                                RhsOpCode>,
//                                WeakForms::Operators::BinaryOpCodes::add>
// operator+(const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode> &lhs_op,
//           const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode> &rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = SymbolicOp<LhsOp, LhsOpCode>;
//   using RhsOpType = SymbolicOp<RhsOp, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

//   return OpType(lhs_op, rhs_op);
// }


// /**
//  * @brief Unary op + binary op
//  */
// template <typename LhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
//           typename RhsOp1,
//           typename RhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<
//   WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode>,
//   WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
//   WeakForms::Operators::BinaryOpCodes::add>
// operator+(
//   const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode> &          lhs_op,
//   const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = SymbolicOp<LhsOp, LhsOpCode>;
//   using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

//   return OpType(lhs_op, rhs_op);
// }



// /**
//  * @brief Binary op + unary op
//  */
// template <typename LhsOp1,
//           typename LhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
//           typename RhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<
//   WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
//   WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode>,
//   WeakForms::Operators::BinaryOpCodes::add>
// operator+(
//   const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
//   const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode> &          rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
//   using RhsOpType = SymbolicOp<RhsOp, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

//   return OpType(lhs_op, rhs_op);
// }



// /**
//  * @brief Binary op + binary op
//  */
// template <typename LhsOp1,
//           typename LhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
//           typename RhsOp1,
//           typename RhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<
//   WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
//   WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
//   WeakForms::Operators::BinaryOpCodes::add>
// operator+(
//   const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
//   const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
//   using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::add>;

//   return OpType(lhs_op, rhs_op);
// }


/* ---------------------------- Subtraction ---------------------------- */


/**
 * @brief Unary op - unary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
          typename... LhsOpArgs,
          typename RhsOp,
          enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,
          typename... RhsOpArgs>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,
  WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,
  WeakForms::Operators::BinaryOpCodes::subtract>
operator-(const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>
            &lhs_op,
          const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>
            &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;
  using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op - binary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
          typename... LhsOpArgs,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
  WeakForms::Operators::BinaryOpCodes::subtract>
operator-(
  const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>
    &                                                              lhs_op,
  const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;
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
          enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,
          typename... RhsOpArgs>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
  WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,
  WeakForms::Operators::BinaryOpCodes::subtract>
operator-(
  const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
  const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>
    &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
  using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;
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


// /**
//  * @brief Unary op - unary op
//  */
// template <typename LhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
//           typename RhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<WeakForms::Operators::SymbolicOp<LhsOp,
// LhsOpCode>,
//                                WeakForms::Operators::SymbolicOp<RhsOp,
//                                RhsOpCode>,
//                                WeakForms::Operators::BinaryOpCodes::subtract>
// operator-(const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode> &lhs_op,
//           const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode> &rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = SymbolicOp<LhsOp, LhsOpCode>;
//   using RhsOpType = SymbolicOp<RhsOp, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

//   return OpType(lhs_op, rhs_op);
// }


// /**
//  * @brief Unary op - binary op
//  */
// template <typename LhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
//           typename RhsOp1,
//           typename RhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<
//   WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode>,
//   WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
//   WeakForms::Operators::BinaryOpCodes::subtract>
// operator-(
//   const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode> &          lhs_op,
//   const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = SymbolicOp<LhsOp, LhsOpCode>;
//   using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

//   return OpType(lhs_op, rhs_op);
// }



// /**
//  * @brief Binary op - unary op
//  */
// template <typename LhsOp1,
//           typename LhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
//           typename RhsOp,
//           enum WeakForms::Operators::SymbolicOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<
//   WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
//   WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode>,
//   WeakForms::Operators::BinaryOpCodes::subtract>
// operator-(
//   const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
//   const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode> &          rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
//   using RhsOpType = SymbolicOp<RhsOp, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

//   return OpType(lhs_op, rhs_op);
// }



// /**
//  * @brief Binary op - binary op
//  */
// template <typename LhsOp1,
//           typename LhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes LhsOpCode,
//           typename RhsOp1,
//           typename RhsOp2,
//           enum WeakForms::Operators::BinaryOpCodes RhsOpCode>
// WeakForms::Operators::BinaryOp<
//   WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,
//   WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,
//   WeakForms::Operators::BinaryOpCodes::subtract>
// operator-(
//   const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,
//   const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)
// {
//   using namespace WeakForms;
//   using namespace WeakForms::Operators;

//   using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;
//   using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;
//   using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::subtract>;

//   return OpType(lhs_op, rhs_op);
// }


/* ---------------------------- Multiplication ---------------------------- */


/**
 * @brief Unary op * unary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
          typename... LhsOpArgs,
          typename RhsOp,
          enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,
          typename... RhsOpArgs>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,
  WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>
            &lhs_op,
          const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>
            &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;
  using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;
  using OpType    = BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::multiply>;

  return OpType(lhs_op, rhs_op);
}


/**
 * @brief Unary op * binary op
 */
template <typename LhsOp,
          enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,
          typename... LhsOpArgs,
          typename RhsOp1,
          typename RhsOp2,
          enum WeakForms::Operators::BinaryOpCodes RhsOpCode,
          typename RhsUnderlyingType>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,
  WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>
            &lhs_op,
          const WeakForms::Operators::
            BinaryOp<RhsOp1, RhsOp2, RhsOpCode, RhsUnderlyingType> &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;
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
          enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,
          typename... RhsOpArgs>
WeakForms::Operators::BinaryOp<
  WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType>,
  WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,
  WeakForms::Operators::BinaryOpCodes::multiply>
operator*(const WeakForms::Operators::
            BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType> &lhs_op,
          const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>
            &rhs_op)
{
  using namespace WeakForms;
  using namespace WeakForms::Operators;

  using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode, LhsUnderlyingType>;
  using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;
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
//   template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
//   struct is_test_function<
//     Operators::BinaryOp<TestFunction<dim, spacedim>, OpCode>> :
//     std::true_type
//   {};

//   template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
//   struct is_trial_solution<
//     Operators::BinaryOp<TrialSolution<dim, spacedim>, OpCode>> :
//     std::true_type
//   {};

//   template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
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
  // template <typename... Args>
  // struct is_binary_op<Operators::BinaryOp<Args...>> : std::true_type
  // {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<
    Operators::
      BinaryOp<LhsOp, RhsOp, Operators::BinaryOpCodes::add, UnderlyingType>>
    : std::true_type
  {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<Operators::BinaryOp<LhsOp,
                                          RhsOp,
                                          Operators::BinaryOpCodes::subtract,
                                          UnderlyingType>> : std::true_type
  {};

  template <typename LhsOp, typename RhsOp, typename UnderlyingType>
  struct is_binary_op<Operators::BinaryOp<LhsOp,
                                          RhsOp,
                                          Operators::BinaryOpCodes::multiply,
                                          UnderlyingType>> : std::true_type
  {};  
  
  
  template <typename T>
  struct is_binary_integral_op<
    T,
    typename std::enable_if<
      Operators::internal::is_valid_binary_integral_op<
        typename T::LhsOpType,
        typename T::RhsOpType>::value &&
      // Can only add / subtract symbolic integrals.
      // In the case of multiplication, the only one Op can
      // be a symbolic integral (i.e. scale it by a factor).
      (T::op_code == Operators::BinaryOpCodes::multiply ?
         (!is_unary_integral_op<typename T::LhsOpType>::value &&
          !is_unary_integral_op<typename T::RhsOpType>::value) :
         Operators::internal::has_addition_or_subtraction_op_code<T>::value)>::
      type> : std::true_type
  {};


  template <typename LhsOp,
            typename RhsOp,
            enum Operators::BinaryOpCodes OpCode>
  struct is_field_solution_op<Operators::BinaryOp<LhsOp, RhsOp, OpCode>>
    : std::conditional<(is_field_solution_op<LhsOp>::value &&
                        !is_test_function_or_trial_solution_op<RhsOp>::value) ||
                         (is_field_solution_op<RhsOp>::value &&
                          !is_test_function_or_trial_solution_op<LhsOp>::value),
                       std::true_type,
                       std::false_type>::type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_binary_operators_h
