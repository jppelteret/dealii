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
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>


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
    template <typename NumberType>
    using value_type = NumberType;

    // Return values at all quadrature points
    template <typename NumberType, int dim, int spacedim = dim>
    using function_type = std::function<std::vector<value_type<NumberType>>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names)>;

    // Return value at one quadrature point
    template <typename NumberType, int dim, int spacedim = dim>
    using qp_function_type = std::function<value_type<NumberType>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names,
      const unsigned int                      q_point)>;

    ScalarCacheFunctor(const std::string &symbol_ascii,
                       const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType, int dim, int spacedim = dim>
    auto
    operator()(const function_type<NumberType, dim, spacedim> &function,
               const UpdateFlags update_flags) const;

    template <typename NumberType, int dim, int spacedim = dim>
    auto
    operator()(const qp_function_type<NumberType, dim, spacedim> &qp_function,
               const UpdateFlags update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType, int dim, int spacedim = dim>
    auto
    value(const function_type<NumberType, dim, spacedim> &function,
          const UpdateFlags                               update_flags) const
    {
      return this->operator()<NumberType, dim, spacedim>(function,
                                                         update_flags);
    }

    template <typename NumberType, int dim, int spacedim = dim>
    auto
    value(const qp_function_type<NumberType, dim, spacedim> &qp_function,
          const UpdateFlags                                  update_flags) const
    {
      return this->operator()<NumberType, dim, spacedim>(qp_function,
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

    template <typename NumberType>
    using value_type = Tensor<rank, spacedim, NumberType>;

    // Return values at all quadrature points
    template <typename NumberType, int dim = spacedim>
    using function_type = std::function<std::vector<value_type<NumberType>>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names)>;

    // Return value at one quadrature point
    template <typename NumberType, int dim = spacedim>
    using qp_function_type = std::function<value_type<NumberType>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names,
      const unsigned int                      q_point)>;

    TensorCacheFunctor(const std::string &symbol_ascii,
                       const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType, int dim = spacedim>
    auto
    operator()(const function_type<NumberType, dim> &function,
               const UpdateFlags                     update_flags) const;

    template <typename NumberType, int dim = spacedim>
    auto
    operator()(const qp_function_type<NumberType, dim> &qp_function,
               const UpdateFlags                        update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType, int dim = spacedim>
    auto
    value(const function_type<NumberType, dim> &function,
          const UpdateFlags                     update_flags) const
    {
      return this->operator()<NumberType, dim>(function, update_flags);
    }

    template <typename NumberType, int dim = spacedim>
    auto
    value(const qp_function_type<NumberType, dim> &qp_function,
          const UpdateFlags                        update_flags) const
    {
      return this->operator()<NumberType, dim>(qp_function, update_flags);
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

    template <typename NumberType>
    using value_type = SymmetricTensor<rank, spacedim, NumberType>;

    // Return values at all quadrature points
    template <typename NumberType, int dim = spacedim>
    using function_type = std::function<std::vector<value_type<NumberType>>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names)>;

    // Return value at one quadrature point
    template <typename NumberType, int dim = spacedim>
    using qp_function_type = std::function<value_type<NumberType>(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &        solution_names,
      const unsigned int                      q_point)>;

    SymmetricTensorCacheFunctor(const std::string &symbol_ascii,
                                const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Call operator to promote this class to a UnaryOp
    template <typename NumberType, int dim = spacedim>
    auto
    operator()(const function_type<NumberType, dim> &function,
               const UpdateFlags                     update_flags) const;

    template <typename NumberType, int dim = spacedim>
    auto
    operator()(const qp_function_type<NumberType, dim> &qp_function,
               const UpdateFlags                        update_flags) const;

    // Let's give our users a nicer syntax to work with this
    // templated call operator.
    template <typename NumberType, int dim = spacedim>
    auto
    value(const function_type<NumberType, dim> &function,
          const UpdateFlags                     update_flags) const
    {
      return this->operator()<NumberType, dim>(function, update_flags);
    }

    template <typename NumberType, int dim = spacedim>
    auto
    value(const qp_function_type<NumberType, dim> &qp_function,
          const UpdateFlags                        update_flags) const
    {
      return this->operator()<NumberType, dim>(qp_function, update_flags);
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
    template <typename NumberType, int dim, int spacedim>
    class UnaryOp<ScalarCacheFunctor,
                  UnaryOpCodes::value,
                  NumberType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarCacheFunctor;

    public:
      using scalar_type = NumberType;

      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType, dim, spacedim>;

      template <typename ResultNumberType = NumberType>
      using qp_function_type =
        typename Op::template qp_function_type<ResultNumberType, dim, spacedim>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int               rank    = 0;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function,
                       const UpdateFlags                update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
      {}

      explicit UnaryOp(const Op &                          operand,
                       const qp_function_type<NumberType> &qp_function,
                       const UpdateFlags                   update_flags)
        : operand(operand)
        , qp_function(qp_function)
        , update_flags(update_flags)
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
        return update_flags;
      }

      /**
       * Return values at all quadrature points
       *
       * This is generic enough that it can operate on cells, faces and
       * subfaces.
       */
      template <typename ResultNumberType = NumberType, int dim2>
      return_type<ResultNumberType>
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

            return_type<NumberType>             out;
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
      const function_type<NumberType>    function;
      const qp_function_type<NumberType> qp_function;
      const UpdateFlags                  update_flags;
    };



    /**
     * Extract the value from a tensor cached functor.
     */
    template <typename NumberType, int dim, int rank_, int spacedim>
    class UnaryOp<TensorCacheFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  NumberType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorCacheFunctor<rank_, spacedim>;

    public:
      using scalar_type = NumberType;

      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType, spacedim>;

      template <typename ResultNumberType = NumberType>
      using qp_function_type =
        typename Op::template qp_function_type<ResultNumberType, spacedim>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int               rank    = rank_;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function,
                       const UpdateFlags                update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
      {}

      explicit UnaryOp(const Op &                          operand,
                       const qp_function_type<NumberType> &qp_function,
                       const UpdateFlags                   update_flags)
        : operand(operand)
        , qp_function(qp_function)
        , update_flags(update_flags)
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
        return update_flags;
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2>
      return_type<ResultNumberType>
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

            return_type<NumberType>             out;
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
      const function_type<NumberType>    function;
      const qp_function_type<NumberType> qp_function;
      const UpdateFlags                  update_flags;
    };



    /**
     * Extract the value from a symmetric tensor cached functor.
     */
    template <typename NumberType, int dim, int rank_, int spacedim>
    class UnaryOp<SymmetricTensorCacheFunctor<rank_, spacedim>,
                  UnaryOpCodes::value,
                  NumberType,
                  WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = SymmetricTensorCacheFunctor<rank_, spacedim>;

    public:
      using scalar_type = NumberType;

      template <typename ResultNumberType = NumberType>
      using value_type = typename Op::template value_type<ResultNumberType>;

      template <typename ResultNumberType = NumberType>
      using function_type =
        typename Op::template function_type<ResultNumberType, spacedim>;

      template <typename ResultNumberType = NumberType>
      using qp_function_type =
        typename Op::template qp_function_type<ResultNumberType, spacedim>;

      template <typename ResultNumberType = NumberType>
      using return_type = std::vector<value_type<ResultNumberType>>;

      static const int               rank    = rank_;
      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");

      explicit UnaryOp(const Op &                       operand,
                       const function_type<NumberType> &function,
                       const UpdateFlags                update_flags)
        : operand(operand)
        , function(function)
        , update_flags(update_flags)
      {}

      explicit UnaryOp(const Op &                          operand,
                       const qp_function_type<NumberType> &qp_function,
                       const UpdateFlags                   update_flags)
        : operand(operand)
        , qp_function(qp_function)
        , update_flags(update_flags)
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
        return update_flags;
      }

      /**
       * Return values at all quadrature points
       */
      template <typename ResultNumberType = NumberType, int dim2>
      return_type<ResultNumberType>
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

            return_type<NumberType>             out;
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
      const function_type<NumberType>    function;
      const qp_function_type<NumberType> qp_function;
      const UpdateFlags                  update_flags;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  template <typename NumberType = double, int dim, int spacedim = dim>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarCacheFunctor,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::ScalarCacheFunctor &operand,
        const typename WeakForms::ScalarCacheFunctor::
          template function_type<NumberType, dim, spacedim> &function,
        const UpdateFlags                                    update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarCacheFunctor;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename NumberType = double, int dim, int spacedim = dim>
  WeakForms::Operators::UnaryOp<WeakForms::ScalarCacheFunctor,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::ScalarCacheFunctor &operand,
        const typename WeakForms::ScalarCacheFunctor::
          template qp_function_type<NumberType, dim, spacedim> &qp_function,
        const UpdateFlags                                       update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarCacheFunctor;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, qp_function, update_flags);
  }



  template <typename NumberType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TensorCacheFunctor<rank, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::TensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
          template function_type<NumberType, dim> &function,
        const UpdateFlags                          update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorCacheFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename NumberType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TensorCacheFunctor<rank, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value,
                                NumberType,
                                internal::DimPack<dim, spacedim>>
  value(const WeakForms::TensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
          template qp_function_type<NumberType, dim> &qp_function,
        const UpdateFlags                             update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorCacheFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, qp_function, update_flags);
  }



  template <typename NumberType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    NumberType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::SymmetricTensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
          template function_type<NumberType, dim> &function,
        const UpdateFlags                          update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorCacheFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function, update_flags);
  }



  template <typename NumberType = double, int dim, int rank, int spacedim>
  WeakForms::Operators::UnaryOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    NumberType,
    internal::DimPack<dim, spacedim>>
  value(const WeakForms::SymmetricTensorCacheFunctor<rank, spacedim> &operand,
        const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
          template qp_function_type<NumberType, dim> &qp_function,
        const UpdateFlags                             update_flags)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorCacheFunctor<rank, spacedim>;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           NumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, qp_function, update_flags);
  }

} // namespace WeakForms



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename NumberType, int dim, int spacedim>
  auto
  ScalarCacheFunctor::
  operator()(const typename WeakForms::ScalarCacheFunctor::
               template function_type<NumberType, dim, spacedim> &function,
             const UpdateFlags update_flags) const
  {
    return WeakForms::value<NumberType>(*this, function, update_flags);
  }


  template <typename NumberType, int dim, int spacedim>
  auto
  ScalarCacheFunctor::operator()(
    const typename WeakForms::ScalarCacheFunctor::
      template qp_function_type<NumberType, dim, spacedim> &qp_function,
    const UpdateFlags                                       update_flags) const
  {
    return WeakForms::value<NumberType>(*this, qp_function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename NumberType, int dim>
  auto
  TensorCacheFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
               template function_type<NumberType, dim> &function,
             const UpdateFlags                          update_flags) const
  {
    return WeakForms::value<NumberType, dim>(*this, function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename NumberType, int dim>
  auto
  TensorCacheFunctor<rank, spacedim>::
  operator()(const typename WeakForms::TensorCacheFunctor<rank, spacedim>::
               template qp_function_type<NumberType, dim> &qp_function,
             const UpdateFlags                             update_flags) const
  {
    return WeakForms::value<NumberType, dim>(*this, qp_function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename NumberType, int dim>
  auto
  SymmetricTensorCacheFunctor<rank, spacedim>::operator()(
    const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
      template function_type<NumberType, dim> &function,
    const UpdateFlags                          update_flags) const
  {
    return WeakForms::value<NumberType, dim>(*this, function, update_flags);
  }


  template <int rank, int spacedim>
  template <typename NumberType, int dim>
  auto
  SymmetricTensorCacheFunctor<rank, spacedim>::operator()(
    const typename WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>::
      template qp_function_type<NumberType, dim> &qp_function,
    const UpdateFlags                             update_flags) const
  {
    return WeakForms::value<NumberType, dim>(*this, qp_function, update_flags);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Unary operations
  template <typename NumberType, int dim, int spacedim>
  struct is_cache_functor<
    WeakForms::Operators::UnaryOp<WeakForms::ScalarCacheFunctor,
                                  WeakForms::Operators::UnaryOpCodes::value,
                                  NumberType,
                                  internal::DimPack<dim, spacedim>>>
    : std::true_type
  {};

  template <typename NumberType, int dim, int rank, int spacedim>
  struct is_cache_functor<
    WeakForms::Operators::UnaryOp<WeakForms::TensorCacheFunctor<rank, spacedim>,
                                  WeakForms::Operators::UnaryOpCodes::value,
                                  NumberType,
                                  internal::DimPack<dim, spacedim>>>
    : std::true_type
  {};

  template <typename NumberType, int dim, int rank, int spacedim>
  struct is_cache_functor<WeakForms::Operators::UnaryOp<
    WeakForms::SymmetricTensorCacheFunctor<rank, spacedim>,
    WeakForms::Operators::UnaryOpCodes::value,
    NumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_cache_functors_h
