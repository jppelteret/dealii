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

#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/operators.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/type_traits.h>



DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    /* ===== Finite element spaces: Test functions and trial solutions ===== */


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

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
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
      const Op &operand;
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

      static const int rank = value_type<double>::rank;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::gradient;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.gradient,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.gradient,
                                                      operand.as_latex());
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
      const Op &operand;
    };


    // All test functions have the same operations as the FE space itself
    template <int dim, int spacedim, enum UnaryOpCodes OpCode>
    class UnaryOp<TestFunction<dim, spacedim>, OpCode>
      : public UnaryOp<Space<dim, spacedim>, OpCode> {
        using Op     = TestFunction<dim, spacedim>;
        using Base_t = UnaryOp<Space<dim, spacedim>, OpCode>;
        public:

          explicit UnaryOp(const Op &operand): Base_t(operand){}
      };


    // All trial solution have the same operations as the FE space itself
    template <int dim, int spacedim, enum UnaryOpCodes OpCode>
    class UnaryOp<TrialSolution<dim, spacedim>, OpCode>
      : public UnaryOp<Space<dim, spacedim>, OpCode> {
        using Op     = TrialSolution<dim, spacedim>;
        using Base_t = UnaryOp<Space<dim, spacedim>, OpCode>;
        public:

          explicit UnaryOp(const Op &operand): Base_t(operand){}
      };



    /* ============== Finite element spaces: Solution fields ============== */


    /**
     * Extract the solution values from the disretised solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<FieldSolution<dim, spacedim>, UnaryOpCodes::value>
    {
      using Op = FieldSolution<dim, spacedim>;

    public:
      template <typename NumberType>
      using value_type = typename Op::template value_type<NumberType>;

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank = 0;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
      }

      // Return solution gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values.get_function_values(solution, out);
        return out;
      }

    private:
      const Op &                     operand;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;
    };



    /**
     * Extract the solution gradients from the disretised solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<FieldSolution<dim, spacedim>, UnaryOpCodes::gradient>
    {
      using Op = FieldSolution<dim, spacedim>;

    public:
      template <typename NumberType>
      using value_type = typename Op::template gradient_type<NumberType>;

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank = value_type<double>::rank;

      explicit UnaryOp(const Op &operand)
        : operand(operand)
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.gradient,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.gradient,
                                                      operand.as_latex());
      }

      // Return solution gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values.get_function_gradients(solution, out);
        return out;
      }

    private:
      const Op &                     operand;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::gradient;
    };



    /* ========================== Functors: Custom ========================== */


    // /**
    //  * Extract the value from a scalar functor.
    //  */
    // template <typename NumberType>
    // class UnaryOp<Functor<NumberType>, UnaryOpCodes::value>
    // {
    //   using Op = Functor<NumberType>;

    // public:
    //   template <typename NumberType2>
    //   using value_type = typename Op::template value_type<NumberType2>;

    //   template <typename NumberType2>
    //   using return_type = std::vector<value_type<NumberType2>>;

    //   static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

    //   explicit UnaryOp(const Op &operand)
    //     : operand(operand)
    //   {}

    //   std::string
    //   as_ascii() const
    //   {
    //     const auto &decorator = operand.get_decorator();
    //     const auto &naming = operand.get_naming_ascii();
    //     return decorator.decorate_with_operator_ascii(
    //       naming.value, operand.as_ascii());
    //   }

    //   std::string
    //   as_latex() const
    //   {
    //     const auto &decorator = operand.get_decorator();
    //     const auto &naming = operand.get_naming_latex();
    //     return decorator.decorate_with_operator_latex(
    //       naming.value, operand.as_latex());
    //   }

    //   // Return single entry
    //   template <typename NumberType2>
    //   const value_type<NumberType2> &
    //   operator()(const unsigned int q_point) const
    //   {
    //     // Should use one of the other [Scalar,Tensor,...]Functors instead.
    //     AssertThrow(false, ExcNotImplemented());

    //     return value_type<NumberType2>{};
    //   }

    //   /**
    //    * Return values at all quadrature points
    //    */
    //   template <typename NumberType2, int dim, int spacedim>
    //   return_type<NumberType2>
    //   operator()(const FEValuesBase<dim, spacedim> &fe_values) const
    //   {
    //     // Should use one of the other [Scalar,Tensor,...]Functors instead.
    //     AssertThrow(false, ExcNotImplemented());

    //     return_type<NumberType> out;
    //     out.reserve(fe_values.n_quadrature_points);

    //     return out;
    //   }

    // private:
    //   const Op &                     operand;
    // };



    /**
     * Extract the value from a scalar functor.
     */
    template <typename NumberType>
    class UnaryOp<ScalarFunctor, UnaryOpCodes::value, NumberType>
    {
      using Op = ScalarFunctor;

    public:
      template <typename NumberType2 = NumberType>
      using value_type = typename Op::template value_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using function_type = typename Op::template function_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using return_type = std::vector<value_type<NumberType2>>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>{}; })
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
      }

      // Return single entry
      template <typename NumberType2 = NumberType>
      const value_type<NumberType2> &
      operator()(const unsigned int q_point) const
      {
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename NumberType2 = NumberType, int dim, int spacedim>
      return_type<NumberType2>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        // TODO: Replace with range based loop
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          out.emplace_back(this->operator()<NumberType2>(q_point));

        return out;
      }

    private:
      const Op &                      operand;
      const function_type<NumberType> function;
    };



    /**
     * Extract the value from a tensor functor.
     */
    template <typename NumberType, int rank, int dim>
    class UnaryOp<TensorFunctor<rank, dim>, UnaryOpCodes::value, NumberType>
    {
      using Op = TensorFunctor<rank, dim>;

    public:
      template <typename NumberType2 = NumberType>
      using value_type = typename Op::template value_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using function_type = typename Op::template function_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using return_type = std::vector<value_type<NumberType2>>;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(
        const Op &                       operand,
        const function_type<NumberType> &function /*, const NumberType &dummy*/)
        : operand(operand)
        , function(function)
      {}

      // explicit UnaryOp(const Op &operand, const NumberType &dummy)
      //   : UnaryOp(operand, [](const unsigned int){return
      //   value_type<NumberType>();})
      // {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
      }

      // Return single entry
      // template <typename NumberType2 = NumberType>
      value_type<NumberType>
      operator()(const unsigned int q_point) const
      {
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename NumberType2 = NumberType, int dim2, int spacedim>
      return_type<NumberType2>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        // TODO: Replace with range based loop
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          out.emplace_back(this->operator()<NumberType2>(q_point));

        return out;
      }

    private:
      const Op &                      operand;
      const function_type<NumberType> function;
    };



    /**
     * Extract the value from a symmetric tensor functor.
     */
    template <typename NumberType, int rank, int dim>
    class UnaryOp<SymmetricTensorFunctor<rank, dim>,
                  UnaryOpCodes::value,
                  NumberType>
    {
      using Op = SymmetricTensorFunctor<rank, dim>;

    public:
      template <typename NumberType2 = NumberType>
      using value_type = typename Op::template value_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using function_type = typename Op::template function_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using return_type = std::vector<value_type<NumberType2>>;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>(); })
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
      }

      // Return single entry
      template <typename NumberType2 = NumberType>
      value_type<NumberType2>
      operator()(const unsigned int q_point) const
      {
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename NumberType2 = NumberType, int dim2, int spacedim>
      return_type<NumberType2>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        // TODO: Replace with range based loop
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          out.emplace_back(this->operator()<NumberType2>(q_point));

        return out;
      }

    private:
      const Op &                      operand;
      const function_type<NumberType> function;
    };



    /* ========================= Functors: deal.II ========================= */



    /**
     * Extract the value from a scalar function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename NumberType, int dim>
    class UnaryOp<ScalarFunctionFunctor<dim>, UnaryOpCodes::value, NumberType>
    {
      using Op = ScalarFunctionFunctor<dim>;

    public:
      template <typename NumberType2 = NumberType>
      using value_type = typename Op::template value_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using function_type = typename Op::template function_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using return_type = std::vector<value_type<NumberType2>>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>{}; })
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
      }

      // Return single entry
      template <typename NumberType2 = NumberType>
      value_type<NumberType2>
      operator()(const Point<dim> &p, const unsigned int component = 0) const
      {
        return function.value(p, component);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename NumberType2 = NumberType, int spacedim>
      return_type<NumberType2>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        // TODO: Replace with range based loop
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          out.emplace_back(this->operator()<NumberType2>(q_point));

        return out;
      }

    private:
      const Op &                       operand;
      const function_type<NumberType> &function;
    };



    /**
     * Extract the value from a tensor function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename NumberType, int rank, int dim>
    class UnaryOp<TensorFunctionFunctor<rank, dim>,
                  UnaryOpCodes::value,
                  NumberType>
    {
      using Op = TensorFunctionFunctor<rank, dim>;

    public:
      template <typename NumberType2 = NumberType>
      using value_type = typename Op::template value_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using function_type = typename Op::template function_type<NumberType2>;

      template <typename NumberType2 = NumberType>
      using return_type = std::vector<value_type<NumberType2>>;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>(); })
      {}

      const SymbolicDecorations &
      get_decorator() const
      {
        return operand.get_decorator();
      }

      std::string
      as_ascii() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii());
      }

      std::string
      as_latex() const
      {
        const auto &decorator = operand.get_decorator();
        const auto &naming    = operand.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex());
      }

      // Return single entry
      template <typename NumberType2 = NumberType>
      value_type<NumberType2>
      operator()(const Point<dim> &p) const
      {
        return function.value(p);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename NumberType2 = NumberType, int dim2, int spacedim>
      return_type<NumberType2>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        // TODO: Replace with range based loop
        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
             ++q_point)
          out.emplace_back(this->operator()<NumberType2>(q_point));

        return out;
      }

    private:
      const Op &                       operand;
      const function_type<NumberType> &function;
    };

  } // namespace Operators



  /* =============== Finite element spaces: Test functions =============== */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TestFunction<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TestFunction<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }



  /* =============== Finite element spaces: Trial solutions =============== */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TrialSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TrialSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }



  /* ============== Finite element spaces: Solution fields ============== */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::FieldSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = FieldSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::FieldSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = FieldSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }



  /* ============================== Functors ============================== */


  // template<typename NumberType>
  // WeakForms::Operators::UnaryOp<WeakForms::Functor<NumberType>,
  //                               WeakForms::Operators::UnaryOpCodes::value>
  // value(const WeakForms::Functor<NumberType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = Functor<NumberType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value>;

  //   return OpType(operand);
  // }



  template <typename NumberType = double>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctor,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType>
  value(
    const WeakForms::ScalarFunctor &operand,
    const typename WeakForms::ScalarFunctor::template function_type<NumberType>
      &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctor;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

    return OpType(operand, function);
  }



  template <typename NumberType = double, int rank, int dim>
  WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, dim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType>
  value(const WeakForms::TensorFunctor<rank, dim> &operand,
        const typename WeakForms::TensorFunctor<rank, dim>::
          template function_type<NumberType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctor<rank, dim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

    return OpType(operand, function /*,NumberType()*/);
  }



  template <typename NumberType = double, int rank, int dim>
  WeakForms::Operators::UnaryOp<WeakForms::SymmetricTensorFunctor<rank, dim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType>
  value(const WeakForms::SymmetricTensorFunctor<rank, dim> &operand,
        const typename WeakForms::SymmetricTensorFunctor<rank, dim>::
          template function_type<NumberType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorFunctor<rank, dim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

    return OpType(operand, function);
  }



  template <typename NumberType = double, int dim>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctionFunctor<dim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType>
  value(const WeakForms::ScalarFunctionFunctor<dim> &operand,
        const typename WeakForms::ScalarFunctionFunctor<
          dim>::template function_type<NumberType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctionFunctor<dim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

    return OpType(operand, function);
  }



  template <typename NumberType = double, int rank, int dim>
  WeakForms::Operators::UnaryOp<WeakForms::TensorFunctionFunctor<rank, dim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType>
  value(const WeakForms::TensorFunctionFunctor<rank, dim> &operand,
        const typename WeakForms::TensorFunctionFunctor<rank, dim>::
          template function_type<NumberType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctionFunctor<rank, dim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

    return OpType(operand, function);
  }


} // namespace WeakForms



#ifndef DOXYGEN


namespace WeakForms
{
  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
    Operators::UnaryOp<TestFunction<dim, spacedim>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
    Operators::UnaryOp<TrialSolution<dim, spacedim>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
    Operators::UnaryOp<FieldSolution<dim, spacedim>, OpCode>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_unary_operators_h
