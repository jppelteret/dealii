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
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/unary_operators.h>


DEAL_II_NAMESPACE_OPEN


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

    Functor(const std::string &        symbol_ascii,
            const std::string &        symbol_latex)
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

    template <typename NumberType>
    using function_type =
      std::function<value_type<NumberType>(const unsigned int q_point)>;

    ScalarFunctor(const std::string &        symbol_ascii,
                  const std::string &        symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}
  };



  template <int rank, int dim>
  class TensorFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    template <typename NumberType>
    using value_type = Tensor<rank, dim, NumberType>;

    template <typename NumberType>
    using function_type =
      std::function<value_type<NumberType>(const unsigned int q_point)>;

    TensorFunctor(const std::string &        symbol_ascii,
                  const std::string &        symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}
  };



  template <int dim>
  using VectorFunctor = TensorFunctor<1, dim>;



  template <int rank, int dim>
  class SymmetricTensorFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    template <typename NumberType>
    using value_type = SymmetricTensor<rank, dim, NumberType>;

    template <typename NumberType>
    using function_type =
      std::function<value_type<NumberType>(const unsigned int q_point)>;

    SymmetricTensorFunctor(
      const std::string &        symbol_ascii,
      const std::string &        symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}
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

    ScalarFunctionFunctor(
      const std::string &        symbol_ascii,
      const std::string &        symbol_latex)
      : Base(symbol_ascii,
             symbol_latex)
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
  };


  // Wrap up a tensor dealii::TensorFunction as a functor
  template <int rank, int dim>
  class TensorFunctionFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    template <typename NumberType>
    using function_type = TensorFunction<rank, dim, NumberType>;

    template <typename NumberType>
    using value_type = typename function_type<NumberType>::value_type;

    template <typename ResultNumberType>
    using gradient_type =
      typename function_type<ResultNumberType>::gradient_type;

    TensorFunctionFunctor(
      const std::string &        symbol_ascii,
      const std::string &        symbol_latex,
      const SymbolicDecorations &decorator = SymbolicDecorations())
      : Base(symbol_ascii,
             symbol_latex)
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
  };



  template <int dim>
  using VectorFunctionFunctor = TensorFunctionFunctor<1, dim>;

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
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
    //   template <typename ResultNumberType, int dim, int spacedim>
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
    template <typename NumberType>
    class UnaryOp<ScalarFunctor, UnaryOpCodes::value, NumberType>
    {
      using Op = ScalarFunctor;

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

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>{}; })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const unsigned int q_point) const
      {
        Assert(function, ExcNotInitialized());
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim, int spacedim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(this->operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<NumberType> function;
    };



    /**
     * Extract the value from a tensor functor.
     */
    template <typename NumberType, int rank_, int dim>
    class UnaryOp<TensorFunctor<rank_, dim>, UnaryOpCodes::value, NumberType>
    {
      using Op = TensorFunctor<rank_, dim>;

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

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const unsigned int q_point) const
      {
        Assert(function, ExcNotInitialized());
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2, int spacedim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<ResultNumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(this->operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<NumberType> function;
    };



    /**
     * Extract the value from a symmetric tensor functor.
     */
    template <typename NumberType, int rank_, int dim>
    class UnaryOp<SymmetricTensorFunctor<rank_, dim>,
                  UnaryOpCodes::value,
                  NumberType>
    {
      using Op = SymmetricTensorFunctor<rank_, dim>;

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

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>(); })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_default;
      }

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const unsigned int q_point) const
      {
        Assert(function, ExcNotInitialized());
        return function(q_point);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2, int spacedim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(this->operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<NumberType> function;
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
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>{}; })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_quadrature_points;
      }

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const Point<dim> &p, const unsigned int component = 0) const
      {
        return function.value(p, component);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int spacedim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.get_quadrature_points())
          out.emplace_back(this->operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                         operand;
      const function_type<NumberType> &function;
    };



    /**
     * Extract the value from a tensor function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename NumberType, int rank_, int dim>
    class UnaryOp<TensorFunctionFunctor<rank_, dim>,
                  UnaryOpCodes::value,
                  NumberType>
    {
      using Op = TensorFunctionFunctor<rank_, dim>;

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
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<NumberType>(); })
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming    = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      // =======

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_quadrature_points;
      }

      // Return single entry
      template <typename ResultNumberType = NumberType>
      value_type<ResultNumberType>
      operator()(const Point<dim> &p) const
      {
        return function.value(p);
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2, int spacedim>
      return_type<ResultNumberType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.get_quadrature_points())
          out.emplace_back(this->operator()<ResultNumberType>(q_point));

        return out;
      }

    private:
      const Op                         operand;
      const function_type<NumberType> &function;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
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



/* ==================== Specialization of type traits ==================== */


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_functors_h
