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



  // template <typename ScalarType = double>
  // WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctor,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               ScalarType>
  // value(
  //   const WeakForms::ScalarFunctor &operand,
  //   const typename WeakForms::ScalarFunctor::template
  //   function_type<ScalarType>
  //     &function);

  // template <typename ScalarType = double, int rank, int spacedim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, spacedim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               ScalarType>
  // value(const WeakForms::TensorFunctor<rank, spacedim> &operand,
  //       const typename WeakForms::TensorFunctor<rank, spacedim>::
  //         template function_type<ScalarType> &function);

  // template <typename ScalarType = double, int rank, int spacedim>
  // WeakForms::Operators::UnaryOp<WeakForms::SymmetricTensorFunctor<rank,
  // spacedim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               ScalarType>
  // value(const WeakForms::SymmetricTensorFunctor<rank, spacedim> &operand,
  //       const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
  //         template function_type<ScalarType> &function);

  // template <typename ScalarType = double, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctionFunctor<dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               ScalarType>
  // value(const WeakForms::ScalarFunctionFunctor<dim> &operand,
  //       const typename WeakForms::ScalarFunctionFunctor<
  //         dim>::template function_type<ScalarType> &function);

  // template <typename ScalarType = double, int rank, int spacedim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctionFunctor<rank,
  // spacedim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               ScalarType>
  // value(const WeakForms::TensorFunctionFunctor<rank, spacedim> &operand,
  //       const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
  //         template function_type<ScalarType> &function);

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

    virtual std::string
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

    virtual std::string
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
    template <typename ScalarType>
    using value_type = ScalarType;
    template <typename ScalarType, int dim, int spacedim = dim>
    using function_type = std::function<
      value_type<ScalarType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    ScalarFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    operator()(const function_type<ScalarType, dim, spacedim> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    value(const function_type<ScalarType, dim, spacedim> &function) const
    {
      return this->operator()<ScalarType, dim, spacedim>(function);
    }
  };



  // class ConstantFunctor : public ScalarFunctor
  // {
  //   using Base = ScalarFunctor

  // public:
  //   template <typename ScalarType>
  //   using value_type = typename Base::value_type<ScalarType>;

  //   template <typename ScalarType>
  //   using function_type = typename Base::function_type<ScalarType>;

  //   template <typename ScalarType>
  //   ScalarFunctor(const ScalarType &value)
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

    template <typename ScalarType>
    using value_type = Tensor<rank, spacedim, ScalarType>;

    template <typename ScalarType, int dim = spacedim>
    using function_type = std::function<
      value_type<ScalarType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    TensorFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename ScalarType, int dim = spacedim>
    auto
    operator()(const function_type<ScalarType, dim> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &function) const
    {
      return this->operator()<ScalarType, dim>(function);
    }
  };



  template <int rank, int spacedim>
  class SymmetricTensorFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename ScalarType>
    using value_type = SymmetricTensor<rank, spacedim, ScalarType>;

    template <typename ScalarType, int dim = spacedim>
    using function_type = std::function<
      value_type<ScalarType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    SymmetricTensorFunctor(const std::string &symbol_ascii,
                           const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename ScalarType, int dim = spacedim>
    auto
    operator()(const function_type<ScalarType, dim> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &function) const
    {
      return this->operator()<ScalarType, dim>(function);
    }
  };



  // Wrap up a scalar dealii::FunctionBase as a functor
  template <int dim>
  class ScalarFunctionFunctor : public Functor<0>
  {
    using Base = Functor<0>;

  public:
    template <typename ScalarType>
    using function_type = Function<dim, ScalarType>;

    // template <typename ScalarType>
    // using value_type = typename function_type<ScalarType>::value_type;

    // template <typename ScalarType>
    // using gradient_type = typename function_type<ScalarType>::gradient_type;

    template <typename ScalarType>
    using value_type = ScalarType;

    template <typename ScalarType>
    using gradient_type = Tensor<1, dim, ScalarType>;

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
    template <typename ScalarType>
    auto
    operator()(const function_type<ScalarType> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType>
    auto
    value(const function_type<ScalarType> &function) const
    {
      return this->operator()<ScalarType>(function);
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

    template <typename ScalarType>
    using function_type = TensorFunction<rank, spacedim, ScalarType>;

    template <typename ScalarType>
    using value_type = typename function_type<ScalarType>::value_type;

    template <typename ResultScalarType>
    using gradient_type =
      typename function_type<ResultScalarType>::gradient_type;

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
    template <typename ScalarType>
    auto
    operator()(const function_type<ScalarType> &function) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType>
    auto
    value(const function_type<ScalarType> &function) const
    {
      return this->operator()<ScalarType>(function);
    }
  };


  // TODO: Add coordinate position functor? Return
  // fe_values.get_quadrature_points()[q]



  template <int dim>
  using VectorFunctor = TensorFunctor<1, dim>;

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


    /**
     * Extract the value from a scalar functor.
     */
    template <typename ScalarType, int dim, int spacedim>
    class UnaryOp<ScalarFunctor,
                  UnaryOpCodes::value,
                  ScalarType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarFunctor;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int               rank    = 0;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<ScalarType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(
        const Op &                    operand,
        const value_type<ScalarType> &value = value_type<ScalarType>{})
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
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<ScalarType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(function(fe_values, q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<ScalarType> function;
    };



    /**
     * Extract the value from a tensor functor.
     */
    template <typename ScalarType, int dim, int rank_, int spacedim>
    class UnaryOp<TensorFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  ScalarType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorFunctor<rank_, spacedim>;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, spacedim>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int               rank    = rank_;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      explicit UnaryOp(
        const Op &                       operand,
        const function_type<ScalarType> &function /*, const ScalarType &dummy*/)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(
        const Op &                    operand,
        const value_type<ScalarType> &value = value_type<ScalarType>{})
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

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<ResultScalarType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(function(fe_values, q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<ScalarType> function;
    };



    /**
     * Extract the value from a symmetric tensor functor.
     */
    template <typename ScalarType, int dim, int rank_, int spacedim>
    class UnaryOp<SymmetricTensorFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  ScalarType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = SymmetricTensorFunctor<rank_, spacedim>;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, spacedim>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int               rank    = rank_;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      explicit UnaryOp(const Op &                       operand,
                       const function_type<ScalarType> &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(
        const Op &                    operand,
        const value_type<ScalarType> &value = value_type<ScalarType>{})
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

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(const FEValuesBase<dim2, spacedim> &fe_values) const
      {
        return_type<ScalarType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.quadrature_point_indices())
          out.emplace_back(function(fe_values, q_point));

        return out;
      }

    private:
      const Op                        operand;
      const function_type<ScalarType> function;
    };



    /* ------------------------ Functors: deal.II ------------------------ */



    /**
     * Extract the value from a scalar function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename ScalarType, int dim>
    class UnaryOp<ScalarFunctionFunctor<dim>, UnaryOpCodes::value, ScalarType>
    {
      using Op = ScalarFunctionFunctor<dim>;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int               rank    = 0;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      /**
       * @brief Construct a new Unary Op object
       *
       * @param operand
       * @param function Non-owning, so the passed in @p function_type must have
       * a longer lifetime than this object.
       */
      explicit UnaryOp(const Op &                       operand,
                       const function_type<ScalarType> &function)
        : operand(operand)
        , function(&function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<ScalarType>{}; })
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

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int spacedim = dim>
      return_type<ResultScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<ScalarType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.get_quadrature_points())
          out.emplace_back(
            this->template operator()<ResultScalarType>(q_point));

        return out;
      }

    private:
      const Op                                            operand;
      const SmartPointer<const function_type<ScalarType>> function;

      // Return single entry
      template <typename ResultScalarType>
      value_type<ResultScalarType>
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
    template <typename ScalarType, int rank_, int spacedim>
    class UnaryOp<TensorFunctionFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  ScalarType>
    {
      using Op = TensorFunctionFunctor<rank_, spacedim>;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int               rank    = rank_;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      /**
       * @brief Construct a new Unary Op object
       *
       * @param operand
       * @param function Non-owning, so the passed in @p function_type must have
       * a longer lifetime than this object.
       */
      explicit UnaryOp(const Op &                       operand,
                       const function_type<ScalarType> &function)
        : operand(operand)
        , function(&function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand,
                  [](const unsigned int) { return value_type<ScalarType>(); })
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

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim>
      return_type<ResultScalarType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return_type<ScalarType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &q_point : fe_values.get_quadrature_points())
          out.emplace_back(
            this->template operator()<ResultScalarType>(q_point));

        return out;
      }

    private:
      const Op                                            operand;
      const SmartPointer<const function_type<ScalarType>> function;

      // Return single entry
      template <typename ResultScalarType>
      value_type<ResultScalarType>
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
  template <typename ScalarType = double, int dim, int spacedim = dim>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctor,
                                WeakForms::Operators::UnaryOpCodes::value,
                                ScalarType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::ScalarFunctor &operand,
        const typename WeakForms::ScalarFunctor::
          template function_type<ScalarType, dim, spacedim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctor;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           ScalarType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function);
  }



  template <typename ScalarType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                ScalarType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::TensorFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorFunctor<rank, spacedim>::
          template function_type<ScalarType, dim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           ScalarType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function /*,ScalarType()*/);
  }



  template <typename ScalarType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<
    WeakForms::SymmetricTensorFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::SymmetricTensorFunctor<rank, spacedim> &operand,
        const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
          template function_type<ScalarType, dim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           ScalarType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function);
  }



  template <typename ScalarType = double, int dim>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctionFunctor<dim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                ScalarType>
  value(const WeakForms::ScalarFunctionFunctor<dim> &operand,
        const typename WeakForms::ScalarFunctionFunctor<
          dim>::template function_type<ScalarType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctionFunctor<dim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, ScalarType>;

    return OpType(operand, function);
  }



  template <typename ScalarType = double, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<
    WeakForms::TensorFunctionFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    ScalarType>
  value(const WeakForms::TensorFunctionFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
          template function_type<ScalarType> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctionFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value, ScalarType>;

    return OpType(operand, function);
  }
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename ScalarType, int dim, int spacedim>
  auto
  ScalarFunctor::operator()(
    const typename WeakForms::ScalarFunctor::
      template function_type<ScalarType, dim, spacedim> &function) const
  {
    return WeakForms::value<ScalarType>(*this, function);
  }


  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  auto
  TensorFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorFunctor<rank, spacedim>::
               template function_type<ScalarType, dim> &function) const
  {
    return WeakForms::value<ScalarType, dim>(*this, function);
  }


  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  auto
  SymmetricTensorFunctor<rank, spacedim>::
  operator()(const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
               template function_type<ScalarType, dim> &function) const
  {
    return WeakForms::value<ScalarType, dim>(*this, function);
  }


  template <int dim>
  template <typename ScalarType>
  auto
  ScalarFunctionFunctor<dim>::
  operator()(const typename WeakForms::ScalarFunctionFunctor<
             dim>::template function_type<ScalarType> &function) const
  {
    return WeakForms::value<ScalarType>(*this, function);
  }


  template <int rank, int spacedim>
  template <typename ScalarType>
  auto
  TensorFunctionFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
               template function_type<ScalarType> &function) const
  {
    return WeakForms::value<ScalarType>(*this, function);
  }

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_functors_h
