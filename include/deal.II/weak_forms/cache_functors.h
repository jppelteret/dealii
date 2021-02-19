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

#ifndef dealii_weakforms_cache_functors_h
#define dealii_weakforms_cache_functors_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/solution_storage.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/symbolic_operators.h>
#include <deal.II/weak_forms/type_traits.h>


DEAL_II_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  /* --------------- Cell face and cell subface operators --------------- */

  class ScalarCacheFunctor;

  template <int rank, int spacedim>
  class TensorCacheFunctor;

  template <int rank, int spacedim>
  class SymmetricTensorCacheFunctor;

} // namespace WeakForms

#endif // DOXYGEN


namespace WeakForms
{
  class ScalarCacheFunctor : public Functor<0>
  {
    using Base = Functor<0>;

  public:
    template <typename ScalarType>
    using value_type = ScalarType;

    // Return values at all quadrature points
    template <typename ScalarType, int dim, int spacedim = dim>
    using function_type = std::function<std::vector<value_type<ScalarType>>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names)>;

    // Return value at one quadrature point
    template <typename ScalarType, int dim, int spacedim = dim>
    using qp_function_type = std::function<value_type<ScalarType>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names,
      const unsigned int                      q_point)>;

    ScalarCacheFunctor(const std::string &symbol_ascii,
                       const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a SymbolicOp
    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    operator()(const function_type<ScalarType, dim, spacedim> &function,
               const UpdateFlags update_flags) const;

    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    operator()(const qp_function_type<ScalarType, dim, spacedim> &qp_function,
               const UpdateFlags update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    value(const function_type<ScalarType, dim, spacedim> &function,
          const UpdateFlags                               update_flags) const
    {
      return this->operator()<ScalarType, dim, spacedim>(function,
                                                         update_flags);
    }

    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    value(const qp_function_type<ScalarType, dim, spacedim> &qp_function,
          const UpdateFlags                                  update_flags) const
    {
      return this->operator()<ScalarType, dim, spacedim>(qp_function,
                                                         update_flags);
    }
  };



  template <int rank, int spacedim>
  class TensorCacheFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename ScalarType>
    using value_type = Tensor<rank, spacedim, ScalarType>;

    // Return values at all quadrature points
    template <typename ScalarType, int dim = spacedim>
    using function_type = std::function<std::vector<value_type<ScalarType>>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names)>;

    // Return value at one quadrature point
    template <typename ScalarType, int dim = spacedim>
    using qp_function_type = std::function<value_type<ScalarType>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names,
      const unsigned int                      q_point)>;

    TensorCacheFunctor(const std::string &symbol_ascii,
                       const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a SymbolicOp
    template <typename ScalarType, int dim = spacedim>
    auto
    operator()(const function_type<ScalarType, dim> &function,
               const UpdateFlags                     update_flags) const;

    template <typename ScalarType, int dim = spacedim>
    auto
    operator()(const qp_function_type<ScalarType, dim> &qp_function,
               const UpdateFlags                        update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &function,
          const UpdateFlags                     update_flags) const
    {
      return this->operator()<ScalarType, dim>(function, update_flags);
    }

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const qp_function_type<ScalarType, dim> &qp_function,
          const UpdateFlags                        update_flags) const
    {
      return this->operator()<ScalarType, dim>(qp_function, update_flags);
    }
  };



  template <int rank, int spacedim>
  class SymmetricTensorCacheFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename ScalarType>
    using value_type = SymmetricTensor<rank, spacedim, ScalarType>;

    // Return values at all quadrature points
    template <typename ScalarType, int dim = spacedim>
    using function_type = std::function<std::vector<value_type<ScalarType>>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names)>;

    // Return value at one quadrature point
    template <typename ScalarType, int dim = spacedim>
    using qp_function_type = std::function<value_type<ScalarType>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names,
      const unsigned int                      q_point)>;

    SymmetricTensorCacheFunctor(const std::string &symbol_ascii,
                                const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a SymbolicOp
    template <typename ScalarType, int dim = spacedim>
    auto
    operator()(const function_type<ScalarType, dim> &function,
               const UpdateFlags                     update_flags) const;

    template <typename ScalarType, int dim = spacedim>
    auto
    operator()(const qp_function_type<ScalarType, dim> &qp_function,
               const UpdateFlags                        update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &function,
          const UpdateFlags                     update_flags) const
    {
      return this->operator()<ScalarType, dim>(function, update_flags);
    }

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const qp_function_type<ScalarType, dim> &qp_function,
          const UpdateFlags                        update_flags) const
    {
      return this->operator()<ScalarType, dim>(qp_function, update_flags);
    }
  };



  template <int dim>
  using VectorCacheFunctor = TensorCacheFunctor<1, dim>;

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ------------------------ Functors: Cached ------------------------ */


    /**
     * Extract the value from a scalar cached functor.
     */
    template <typename ScalarType, int dim, int spacedim>
    class SymbolicOp<ScalarCacheFunctor,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarCacheFunctor;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using qp_function_type =
        typename Op::template qp_function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int                  rank    = 0;
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      explicit SymbolicOp(const Op &                       operand,
                          const function_type<ScalarType> &function,
                          const UpdateFlags                update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
      {}

      explicit SymbolicOp(const Op &                          operand,
                          const qp_function_type<ScalarType> &qp_function,
                          const UpdateFlags                   update_flags)
        : operand(operand)
        , qp_function(qp_function)
        , update_flags(update_flags)
      {}

      // Copy constructor so that we can wrap this is a special class that
      // will help deal with the result of differential operations.
      SymbolicOp(const SymbolicOp &symbolic_operand) = default;

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
        return update_flags;
      }

      /**
       * Return values at all quadrature points
       *
       * This is generic enough that it can operate on cells, faces and
       * subfaces.
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        if (function)
          {
            return function(scratch_data, solution_names);
          }
        else
          {
            Assert(qp_function, ExcNotInitialized());

            return_type<ScalarType>             out;
            const FEValuesBase<dim2, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                qp_function(scratch_data, solution_names, q_point));

            return out;
          }
      }

    private:
      const Op                           operand;
      const function_type<ScalarType>    function;
      const qp_function_type<ScalarType> qp_function;
      const UpdateFlags                  update_flags;
    };



    /**
     * Extract the value from a tensor cached functor.
     */
    template <typename ScalarType, int dim, int rank_, int spacedim>
    class SymbolicOp<TensorCacheFunctor<rank_, spacedim>,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorCacheFunctor<rank_, spacedim>;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, spacedim>;

      template <typename ResultScalarType>
      using qp_function_type =
        typename Op::template qp_function_type<ResultScalarType, spacedim>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int                  rank    = rank_;
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      explicit SymbolicOp(const Op &                       operand,
                          const function_type<ScalarType> &function,
                          const UpdateFlags                update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
      {}

      explicit SymbolicOp(const Op &                          operand,
                          const qp_function_type<ScalarType> &qp_function,
                          const UpdateFlags                   update_flags)
        : operand(operand)
        , qp_function(qp_function)
        , update_flags(update_flags)
      {}

      // Copy constructor so that we can wrap this is a special class that
      // will help deal with the result of differential operations.
      SymbolicOp(const SymbolicOp &symbolic_operand) = default;

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
        return update_flags;
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        if (function)
          {
            return function(scratch_data, solution_names);
          }
        else
          {
            Assert(qp_function, ExcNotInitialized());

            return_type<ScalarType>             out;
            const FEValuesBase<dim2, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                qp_function(scratch_data, solution_names, q_point));

            return out;
          }
      }

    private:
      const Op                           operand;
      const function_type<ScalarType>    function;
      const qp_function_type<ScalarType> qp_function;
      const UpdateFlags                  update_flags;
    };



    /**
     * Extract the value from a symmetric tensor cached functor.
     */
    template <typename ScalarType, int dim, int rank_, int spacedim>
    class SymbolicOp<SymmetricTensorCacheFunctor<rank_, spacedim>,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = SymmetricTensorCacheFunctor<rank_, spacedim>;

    public:
      using scalar_type = ScalarType;

      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, spacedim>;

      template <typename ResultScalarType>
      using qp_function_type =
        typename Op::template qp_function_type<ResultScalarType, spacedim>;

      template <typename ResultScalarType>
      using return_type = std::vector<value_type<ResultScalarType>>;

      static const int                  rank    = rank_;
      static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      explicit SymbolicOp(const Op &                       operand,
                          const function_type<ScalarType> &function,
                          const UpdateFlags                update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
      {}

      explicit SymbolicOp(const Op &                          operand,
                          const qp_function_type<ScalarType> &qp_function,
                          const UpdateFlags                   update_flags)
        : operand(operand)
        , qp_function(qp_function)
        , update_flags(update_flags)
      {}

      // Copy constructor so that we can wrap this is a special class that
      // will help deal with the result of differential operations.
      SymbolicOp(const SymbolicOp &symbolic_operand) = default;

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
        return update_flags;
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultScalarType, int dim2>
      return_type<ResultScalarType>
      operator()(MeshWorker::ScratchData<dim2, spacedim> &scratch_data,
                 const std::vector<std::string> &         solution_names) const
      {
        if (function)
          {
            return function(scratch_data, solution_names);
          }
        else
          {
            Assert(qp_function, ExcNotInitialized());

            return_type<ScalarType>             out;
            const FEValuesBase<dim2, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                qp_function(scratch_data, solution_names, q_point));

            return out;
          }
      }

    private:
      const Op                           operand;
      const function_type<ScalarType>    function;
      const qp_function_type<ScalarType> qp_function;
      const UpdateFlags                  update_flags;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename ScalarType = double, int dim, int spacedim = dim>
  WeakForms::Operators::SymbolicOp<WeakForms::ScalarCacheFunctor,
                                   WeakForms::Operators::SymbolicOpCodes::value,
                                   ScalarType,
                                   internal::DimPack<dim, spacedim>>
  value(const WeakForms::ScalarCacheFunctor &operand,
        const typename WeakForms::ScalarCacheFunctor::
          template function_type<ScalarType, dim, spacedim> &function,
        const UpdateFlags                                    update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarCacheFunctor;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename ScalarType = double, int dim, int spacedim = dim>
  WeakForms::Operators::SymbolicOp<WeakForms::ScalarCacheFunctor,
                                   WeakForms::Operators::SymbolicOpCodes::value,
                                   ScalarType,
                                   internal::DimPack<dim, spacedim>>
  value(const WeakForms::ScalarCacheFunctor &operand,
        const typename WeakForms::ScalarCacheFunctor::
          template qp_function_type<ScalarType, dim, spacedim> &qp_function,
        const UpdateFlags                                       update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarCacheFunctor;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, qp_function, update_flags);
  }



  template <typename ScalarType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::TensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
          template function_type<ScalarType, dim> &function,
        const UpdateFlags                          update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorCacheFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename ScalarType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::TensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::TensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
          template qp_function_type<ScalarType, dim> &qp_function,
        const UpdateFlags                             update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorCacheFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, qp_function, update_flags);
  }



  template <typename ScalarType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::SymmetricTensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
          template function_type<ScalarType, dim> &function,
        const UpdateFlags                          update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorCacheFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename ScalarType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::SymbolicOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::SymmetricTensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
          template qp_function_type<ScalarType, dim> &qp_function,
        const UpdateFlags                             update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorCacheFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, qp_function, update_flags);
  }

} // namespace WeakForms



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename ScalarType, int dim, int spacedim>
  auto
  ScalarCacheFunctor::
  operator()(const typename WeakForms::ScalarCacheFunctor::
               template function_type<ScalarType, dim, spacedim> &function,
             const UpdateFlags update_flags) const
  {
    return WeakForms::value<ScalarType>(*this, function, update_flags);
  }


  template <typename ScalarType, int dim, int spacedim>
  auto
  ScalarCacheFunctor::operator()(
    const typename WeakForms::ScalarCacheFunctor::
      template qp_function_type<ScalarType, dim, spacedim> &qp_function,
    const UpdateFlags                                       update_flags) const
  {
    return WeakForms::value<ScalarType>(*this, qp_function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  auto
  TensorCacheFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
               template function_type<ScalarType, dim> &function,
             const UpdateFlags                          update_flags) const
  {
    return WeakForms::value<ScalarType, dim>(*this, function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  auto
  TensorCacheFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
               template qp_function_type<ScalarType, dim> &qp_function,
             const UpdateFlags                             update_flags) const
  {
    return WeakForms::value<ScalarType, dim>(*this, qp_function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  auto
  SymmetricTensorCacheFunctor<rank, spacedim>::operator()(
    const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
      template function_type<ScalarType, dim> &function,
    const UpdateFlags                          update_flags) const
  {
    return WeakForms::value<ScalarType, dim>(*this, function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  auto
  SymmetricTensorCacheFunctor<rank, spacedim>::operator()(
    const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
      template qp_function_type<ScalarType, dim> &qp_function,
    const UpdateFlags                             update_flags) const
  {
    return WeakForms::value<ScalarType, dim>(*this, qp_function, update_flags);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Unary operations
  template <typename ScalarType, int dim, int spacedim>
  struct is_cache_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::ScalarCacheFunctor,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_cache_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::TensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_cache_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  // Unary operations
  template <typename ScalarType, int dim, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::ScalarCacheFunctor,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::TensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_cache_functors_h
