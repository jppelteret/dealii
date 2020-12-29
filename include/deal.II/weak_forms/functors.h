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

#ifndef dealii_weakforms_functors_h
#define dealii_weakforms_functors_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/unary_operators.h>


DEAL_II_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  /* --------------- Cell face and cell subface operators --------------- */

  template <int rank>
  class Functor;

  class ScalarFunctor;

  template <int rank, int spacedim>
  class TensorFunctor;

  template <int rank, int spacedim>
  class SymmetricTensorFunctor;

  template <int spacedim>
  class ScalarFunctionFunctor;

  template <int rank, int spacedim>
  class TensorFunctionFunctor;



  // template <typename NumberType = double>
  // WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctor,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(
  //   const WeakForms::ScalarFunctor &operand,
  //   const typename WeakForms::ScalarFunctor::template
  //   function_type<NumberType>
  //     &function);

  // template <typename NumberType = double, int rank, int spacedim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, spacedim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::TensorFunctor<rank, spacedim> &operand,
  //       const typename WeakForms::TensorFunctor<rank, spacedim>::
  //         template function_type<NumberType> &function);

  // template <typename NumberType = double, int rank, int spacedim>
  // WeakForms::Operators::UnaryOp<WeakForms::SymmetricTensorFunctor<rank,
  // spacedim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::SymmetricTensorFunctor<rank, spacedim> &operand,
  //       const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
  //         template function_type<NumberType> &function);

  // template <typename NumberType = double, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctionFunctor<dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::ScalarFunctionFunctor<dim> &operand,
  //       const typename WeakForms::ScalarFunctionFunctor<
  //         dim>::template function_type<NumberType> &function);

  // template <typename NumberType = double, int rank, int spacedim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctionFunctor<rank,
  // spacedim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::TensorFunctionFunctor<rank, spacedim> &operand,
  //       const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
  //         template function_type<NumberType> &function);

} // namespace WeakForms

#endif // DOXYGEN


namespace WeakForms
{
  // The meat in the middle of the WeakForms
  template <int rank_>
  class Functor
  {
  public:
    /**
     * Rank of this object operates.
     */
    static const unsigned int rank = rank_;

    Functor(const std::string &symbol_ascii, const std::string &symbol_latex)
      : symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
    {}

    virtual ~Functor() = default;

    // ----  Ascii ----

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.unary_op_functor_as_ascii(*this, rank);
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      return symbol_ascii;
    }

    // ---- LaTeX ----

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.unary_op_functor_as_latex(*this, rank);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      return symbol_latex;
    }

  protected:
    const std::string symbol_ascii;
    const std::string symbol_latex;
  };



  class ScalarFunctor : public Functor<0>
  {
    using Base = Functor<0>;

  public:
    template <typename NumberType>
    using value_type = NumberType;
    template <typename NumberType, int dim, int spacedim = dim>
    using function_type = std::function<
      value_type<NumberType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    ScalarFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType, int dim, int spacedim = dim>
    auto
    operator()(const function_type<NumberType, dim, spacedim> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType, int dim, int spacedim = dim>
    auto
    value(const function_type<NumberType, dim, spacedim> &function) const
    {
      return this->operator()<NumberType, dim, spacedim>(function);
    }
  };



  // class ConstantFunctor : public ScalarFunctor
  // {
  //   using Base = ScalarFunctor

  // public:
  //   template <typename NumberType>
  //   using value_type = typename Base::value_type<NumberType>;

  //   template <typename NumberType>
  //   using function_type = typename Base::function_type<NumberType>;

  //   template <typename NumberType>
  //   ScalarFunctor(const NumberType &value)
  //     : Base(Utilities::to_string(value),
  //            Utilities::to_string(value))
  //   {}
  // };



  template <int rank, int spacedim>
  class TensorFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename NumberType>
    using value_type = Tensor<rank, spacedim, NumberType>;

    template <typename NumberType, int dim = spacedim>
    using function_type = std::function<
      value_type<NumberType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    TensorFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType, int dim = spacedim>
    auto
    operator()(const function_type<NumberType, dim> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType, int dim = spacedim>
    auto
    value(const function_type<NumberType, dim> &function) const
    {
      return this->operator()<NumberType, dim>(function);
    }
  };



  template <int dim>
  using VectorFunctor = TensorFunctor<1, dim>;



  template <int rank, int spacedim>
  class SymmetricTensorFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename NumberType>
    using value_type = SymmetricTensor<rank, spacedim, NumberType>;

    template <typename NumberType, int dim = spacedim>
    using function_type = std::function<
      value_type<NumberType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    SymmetricTensorFunctor(const std::string &symbol_ascii,
                           const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType, int dim = spacedim>
    auto
    operator()(const function_type<NumberType, dim> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType, int dim = spacedim>
    auto
    value(const function_type<NumberType, dim> &function) const
    {
      return this->operator()<NumberType, dim>(function);
    }
  };



  // Wrap up a scalar dealii::FunctionBase as a functor
  template <int dim>
  class ScalarFunctionFunctor : public Functor<0>
  {
    using Base = Functor<0>;

  public:
    template <typename NumberType>
    using function_type = Function<dim, NumberType>;

    // template <typename NumberType>
    // using value_type = typename function_type<NumberType>::value_type;

    // template <typename NumberType>
    // using gradient_type = typename function_type<NumberType>::gradient_type;

    template <typename NumberType>
    using value_type = NumberType;

    template <typename NumberType>
    using gradient_type = Tensor<1, dim, NumberType>;

    ScalarFunctionFunctor(const std::string &symbol_ascii,
                          const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_ascii(this->symbol_ascii);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_latex(this->symbol_latex);
    }

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType>
    auto
    operator()(const function_type<NumberType> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType>
    auto
    value(const function_type<NumberType> &function) const
    {
      return this->operator()<NumberType>(function);
    }
  };


  // Wrap up a tensor dealii::TensorFunction as a functor
  template <int rank, int spacedim>
  class TensorFunctionFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename NumberType>
    using function_type = TensorFunction<rank, spacedim, NumberType>;

    template <typename NumberType>
    using value_type = typename function_type<NumberType>::value_type;

    template <typename ResultNumberType>
    using gradient_type =
      typename function_type<ResultNumberType>::gradient_type;

    TensorFunctionFunctor(
      const std::string &        symbol_ascii,
      const std::string &        symbol_latex,
      const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii, symbol_latex)
    {}

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_ascii(this->symbol_ascii);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_latex(this->symbol_latex);
    }

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType>
    auto
    operator()(const function_type<NumberType> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType>
    auto
    value(const function_type<NumberType> &function) const
    {
      return this->operator()<NumberType>(function);
    }
  };



  template <int dim>
  using VectorFunctionFunctor = TensorFunctionFunctor<1, dim>;

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace internal
  {
    // Used to work around the restriction that template arguments
    // for template type parameter must be a type
    template <int dim_, int spacedim_>
    struct DimPack
    {
      static const unsigned int dim      = dim_;
      static const unsigned int spacedim = spacedim_;
    };
  } // namespace internal

  namespace Operators
  {
    /* ------------------------ Functors: Custom ------------------------ */


    // /**
    //  * Extract the value from a scalar functor.
    //  */
    // template <typename NumberType>
    // class UnaryOp<Functor<NumberType>, UnaryOpCodes::value>
    // {
    //   using Op = Functor<NumberType>;

    // public:
    //   template <typename ResultNumberType>
    //   using value_type = typename Op::template value_type<ResultNumberType>;

    //   template <typename ResultNumberType>
    //   using return_type = std::vector<value_type<ResultNumberType>>;

    //   static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

    //   explicit UnaryOp(const Op &operand)
    //     : operand(operand)
    //   {}

    //   std::string
    //   as_ascii(const SymbolicDecorations &decorator) const
    //   {
    //     const auto &naming = decorator.get_naming_ascii();
    //     return decorator.decorate_with_operator_ascii(
    //       naming.value, operand.as_ascii(decorator));
    //   }

    //   std::string
    //   as_latex(const SymbolicDecorations &decorator) const
    //   {
    //     const auto &naming = decorator.get_naming_latex();
    //     return decorator.decorate_with_operator_latex(
    //       naming.value, operand.as_latex(decorator));
    //   }

    //   // Return single entry
    //   template <typename ResultNumberType>
    //   value_type<ResultNumberType>
    //   operator()(const unsigned int q_point) const
    //   {
    //     // Should use one of the other [Scalar,Tensor,...]Functors instead.
    //     AssertThrow(false, ExcNotImplemented());

    //     return value_type<ResultNumberType>{};
    //   }

    //   /**
    //    * Return values at all quadrature points
    //    */
    //   template <typename ResultNumberType, int dim, int spacedim = dim>
    //   return_type<ResultNumberType>
    //   operator()(const FEValuesBase<dim, spacedim> &fe_values) const
    //   {
    //     // Should use one of the other [Scalar,Tensor,...]Functors instead.
    //     AssertThrow(false, ExcNotImplemented());

    //     return_type<NumberType> out;
    //     out.reserve(fe_values.n_quadrature_points);

    //     return out;
    //   }

    // private:
    //   const Op                       operand;
    // };



    /**
     * Extract the value from a scalar functor.
     */
    template <typename NumberType, int dim, int spacedim>
    class UnaryOp<ScalarFunctor,
                  UnaryOpCodes::value,
                  NumberType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarFunctor;

    public:
      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType, dim, spacedim>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(
        const Op &                    operand,
        const value_type<NumberType> &value = value_type<NumberType>{})
        : UnaryOp(operand,
                  [value](const FEValuesBase<dim, spacedim> &fe_values,
                          const unsigned int) { return value; })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      // template <ypename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const FEValuesBase<dim, spacedim> &fe_values,
      //            const unsigned int                 q_point) const
      // {
      //   (void)fe_values;
      //   return function(q_point);
      // }

      /**
       * Return values at all quadrature points
       *
       * This is generic enough that it can operate on cells, faces and
       * subfaces. The user can cast the @p fe_values values object into
       * the base type for face values if necessary. The user can get the
       * current cell by a call to `fe_values.get_cell()` and, if cast to
       * an FEFaceValuesBase type, then `fe_face_values.get_face_index()`
       * returns the face index.
       */
      template <typename ResultNumberType = NumberType, int dim2>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(function(fe_values, q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<NumberType> function;

      // // Return single entry
      // template <typename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const unsigned int q_point) const
      // {
      //   Assert(function, ExcNotInitialized());
      //   return function(q_point);
      // }
    };



    /**
     * Extract the value from a tensor functor.
     */
    template <typename NumberType, int dim, int rank_, int spacedim>
    class UnaryOp<TensorFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  NumberType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorFunctor<rank_, spacedim>;

    public:
      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType, spacedim>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int rank = rank_;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(
        const Op &                       operand,
        const function_type<NumberType> &function /*, const NumberType &dummy*/)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(
        const Op &                    operand,
        const value_type<NumberType> &value = value_type<NumberType>{})
        : UnaryOp(operand,
                  [value](const FEValuesBase<dim, spacedim> &fe_values,
                          const unsigned int) { return value; })
      {}

      // explicit UnaryOp(const Op &operand, const NumberType &dummy)
      //   : UnaryOp(operand, [](const unsigned int){return
      //   value_type<NumberType>();})
      // {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      // template <ypename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const FEValuesBase<dim, spacedim> &fe_values,
      //            const unsigned int                 q_point) const
      // {
      //   (void)fe_values;
      //   return function(q_point);
      // }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<ResultNumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(function(fe_values, q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<NumberType> function;

      // // Return single entry
      // template <typename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const unsigned int q_point) const
      // {
      //   Assert(function, ExcNotInitialized());
      //   return function(q_point);
      // }
    };



    /**
     * Extract the value from a symmetric tensor functor.
     */
    template <typename NumberType, int dim, int rank_, int spacedim>
    class UnaryOp<SymmetricTensorFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  NumberType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = SymmetricTensorFunctor<rank_, spacedim>;

    public:
      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType, spacedim>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int rank = rank_;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(
        const Op &                    operand,
        const value_type<NumberType> &value = value_type<NumberType>{})
        : UnaryOp(operand,
                  [value](const FEValuesBase<dim, spacedim> &fe_values,
                          const unsigned int) { return value; })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      // template <ypename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const FEValuesBase<dim, spacedim> &fe_values,
      //            const unsigned int                 q_point) const
      // {
      //   (void)fe_values;
      //   return function(q_point);
      // }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(function(fe_values, q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<NumberType> function;

      // // Return single entry
      // template <typename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const unsigned int q_point) const
      // {
      //   Assert(function, ExcNotInitialized());
      //   return function(q_point);
      // }
    };



    /* ------------------------ Functors: deal.II ------------------------ */



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
      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      /**
       * @brief Construct a new Unary Op object
       *
       * @param operand
       * @param function Non-owning, so the passed in @p function_type must have
       * a longer lifetime than this object.
       */
      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(&function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>{}; })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_quadrature_points;
      }

      // template <ypename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const FEValuesBase<dim, spacedim> &fe_values,
      //            const unsigned int                 q_point,
      //            const unsigned int                 component = 0)) const
      // {
      //   return function.value(fe_values.quadrature_point(q_point),
      //   component);
      // }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int spacedim = dim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.get_quadrature_points())
          out.emplace_back(
            this->template operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                                            operand;
      const SmartPointer<const function_type<NumberType>> function;

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const Point<dim> &p, const unsigned int component = 0) const
      {
        return function->value(p, component);
      }
    };



    /**
     * Extract the value from a tensor function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename NumberType, int rank_, int spacedim>
    class UnaryOp<TensorFunctionFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  NumberType>
    {
      using Op = TensorFunctionFunctor<rank_, spacedim>;

    public:
      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int rank = rank_;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      /**
       * @brief Construct a new Unary Op object
       *
       * @param operand
       * @param function Non-owning, so the passed in @p function_type must have
       * a longer lifetime than this object.
       */
      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(&function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>(); })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(
          naming.value, operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(
          naming.value, operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_quadrature_points;
      }

      // template <ypename ResultNumberType = NumberType>
      // value_type<ResultNumberType>
      // operator()(const FEValuesBase<dim, spacedim> &fe_values,
      //            const unsigned int                 q_point,
      //            const unsigned int                 component = 0)) const
      // {
      //   Assert(component ==0, ExcMessage("Vector-valued functions with
      //   several components are not supported.")); return
      //   function.value(fe_values.quadrature_point(q_point));
      // }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.get_quadrature_points())
          out.emplace_back(
            this->template operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                                            operand;
      const SmartPointer<const function_type<NumberType>> function;

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const Point<spacedim> &p) const
      {
        return function->value(p);
      }
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename NumberType = double, int dim, int spacedim = dim>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctor,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::ScalarFunctor &operand,
        const typename WeakForms::ScalarFunctor::
          template function_type<NumberType, dim, spacedim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctor;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function);
  }



  template <typename NumberType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::TensorFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorFunctor<rank, spacedim>::
          template function_type<NumberType, dim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function /*,NumberType()*/);
  }



  template <typename NumberType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<
    WeakForms::SymmetricTensorFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    NumberType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::SymmetricTensorFunctor<rank, spacedim> &operand,
        const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
          template function_type<NumberType, dim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

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



  template <typename NumberType = double, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<
    WeakForms::TensorFunctionFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    NumberType>
  value(const WeakForms::TensorFunctionFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
          template function_type<NumberType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctionFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

    return OpType(operand, function);
  }
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename NumberType, int dim, int spacedim>
  auto
  ScalarFunctor::operator()(
    const typename WeakForms::ScalarFunctor::
      template function_type<NumberType, dim, spacedim> &function) const
  {
    return WeakForms::value<NumberType>(*this, function);
  }


  template <int rank, int spacedim>
  template <typename NumberType, int dim>
  auto
  TensorFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorFunctor<rank, spacedim>::
               template function_type<NumberType, dim> &function) const
  {
    return WeakForms::value<NumberType, dim>(*this, function);
  }


  template <int rank, int spacedim>
  template <typename NumberType, int dim>
  auto
  SymmetricTensorFunctor<rank, spacedim>::
  operator()(const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
               template function_type<NumberType, dim> &function) const
  {
    return WeakForms::value<NumberType, dim>(*this, function);
  }


  template <int dim>
  template <typename NumberType>
  auto
  ScalarFunctionFunctor<dim>::
  operator()(const typename WeakForms::ScalarFunctionFunctor<
             dim>::template function_type<NumberType> &function) const
  {
    return WeakForms::value<NumberType>(*this, function);
  }


  template <int rank, int spacedim>
  template <typename NumberType>
  auto
  TensorFunctionFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
               template function_type<NumberType> &function) const
  {
    return WeakForms::value<NumberType>(*this, function);
  }

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_functors_h
