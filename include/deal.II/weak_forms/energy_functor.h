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

#ifndef dealii_weakforms_energy_functor_h
#define dealii_weakforms_energy_functor_h

#include <deal.II/base/config.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/weak_forms/functors.h>
#include <deal.II/weak_forms/solution_storage.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename... UnaryOpsSubSpaceFieldSolution>
  class EnergyFunctor : public WeakForms::Functor<0>
  {
    using Base = WeakForms::Functor<0>;

  public:
    template <typename NumberType>
    using value_type = NumberType;

    template <typename NumberType, int dim, int spacedim = dim>
    using function_type = std::function<value_type<NumberType>(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const std::vector<std::string> &              solution_names,
      const unsigned int                            q_point,
      const typename UnaryOpsSubSpaceFieldSolution::template value_type<
        NumberType> &... field_solutions)>;

    template <typename ScalarType,
              enum Differentiation::AD::NumberTypes ADNumberTypeCode>
    using ad_type =
      typename Differentiation::AD::NumberTraits<ScalarType,
                                                 ADNumberTypeCode>::ad_type;

    EnergyFunctor(
      const std::string &symbol_ascii,
      const std::string &symbol_latex,
      const UnaryOpsSubSpaceFieldSolution &... unary_op_field_solutions)
      : Base(symbol_ascii, symbol_latex)
      , unary_op_field_solutions(unary_op_field_solutions...)
    {}

    // ----  Ascii ----

    virtual std::string
    as_ascii(const SymbolicDecorations &decorator) const override
    {
      return Base::as_ascii(decorator) +
             decorator.unary_field_ops_as_ascii(get_field_args());
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return symbol_ascii;
    }

    // ---- LaTeX ----

    virtual std::string
    as_latex(const SymbolicDecorations &decorator) const override
    {
      return Base::as_latex(decorator) +
             decorator.unary_field_ops_as_latex(get_field_args());
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return symbol_latex;
    }

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

  private:
    const std::tuple<UnaryOpsSubSpaceFieldSolution...> unary_op_field_solutions;

    const std::tuple<UnaryOpsSubSpaceFieldSolution...> &
    get_field_args() const
    {
      return unary_op_field_solutions;
    }
  };

  // namespace AutoDifferentiation
  // {

  //   template <int dim>
  //   using VectorFunctor = WeakForms::VectorFunctor<dim>;

  //   template <int rank, int dim>
  //   using TensorFunctor = WeakForms::TensorFunctor<rank, dim>;

  //   template <int rank, int dim>
  //   using SymmetricTensorFunctor = WeakForms::SymmetricTensorFunctor<rank,
  //   dim>;

  //   template <int dim>
  //   using ScalarFunctionFunctor = WeakForms::ScalarFunctionFunctor<dim>;

  //   template <int rank, int dim>
  //   using TensorFunctionFunctor = WeakForms::TensorFunctionFunctor<rank,
  //   dim>;

  // } // namespace AutoDifferentiation

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ------------------------ Functors: Custom ------------------------ */

    /**
     * Extract the value from a scalar functor.
     */
    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename... UnaryOpsSubSpaceFieldSolution>
    class UnaryOp<
      EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
      UnaryOpCodes::value,
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
      ADNumberType,
      WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;

    public:
      using scalar_type =
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;

      static constexpr enum Differentiation::AD::NumberTypes ADNumberTypeCode =
        Differentiation::AD::ADNumberTraits<ADNumberType>::type_code;

      using ADHelperType = Differentiation::AD::
        ScalarFunction<spacedim, ADNumberTypeCode, scalar_type>;
      using ad_type =
        typename Differentiation::AD::NumberTraits<scalar_type,
                                                   ADNumberTypeCode>::ad_type;
      static_assert(
        std::is_same<typename ADHelperType::ad_type, ADNumberType>::value,
        "AD types not the same.");
      static_assert(std::is_same<ad_type, ADNumberType>::value,
                    "AD types not the same.");

      template <typename ResultNumberType = ad_type>
      using value_type = typename Op::template value_type<ResultNumberType>;

      // template <typename ResultNumberType = ad_type>
      //   using functor_arguments =
      //     std::tuple<typename UnaryOpsSubSpaceFieldSolution::
      //                           template value_type<ResultNumberType>...>;

      template <typename ResultNumberType = ad_type>
      using function_type =
        typename Op::template function_type<ResultNumberType, dim, spacedim>;

      template <typename ResultNumberType = ad_type>
      using return_type = std::vector<value_type<ResultNumberType>>;

      using ad_function_type = function_type<ad_type>;

      static const int rank = 0;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      explicit UnaryOp(const Op &operand, const ad_function_type &function)
        : operand(operand)
        , function(function)
      {}

      explicit UnaryOp(const Op &operand)
        : UnaryOp(operand, [](const unsigned int) { return ad_type{}; })
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
      template <typename ResultNumberType = ad_type, int dim2>
      return_type<ResultNumberType>
      operator()(const MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &solution_names) const
      {
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.get_current_fe_values();
        // Don't trust the AD number type to initialize itself properly within
        // a vector.
        return_type<ResultNumberType> out(fe_values.n_quadrature_points,
                                          ResultNumberType());

        for (const auto &q_point : fe_values.quadrature_point_indices())
          {
            // TODO: The values passed to the function should come from the
            // ADHelper...
            out[q_point] =
              function(scratch_data,
                       solution_names,
                       q_point,
                       UnaryOpsSubSpaceFieldSolution::template value_type<
                         ResultNumberType>()...);
          }

        AssertThrow(false, ExcMessage("Not yet implemented."));

        return out;
      }

    private:
      const Op               operand;
      const ad_function_type function;
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * Shortcut so that we don't need to do something like this:
   *
   * <code>
   * const FieldSolution<dim> solution;
   * const WeakForms::SubSpaceExtractors::Scalar subspace_extractor(0, "s",
   * "s");
   *
   * const auto soln_ss   = solution[subspace_extractor];
   * const auto soln_val  = soln_ss.value();    // Solution value
   * const auto soln_grad = soln_ss.gradient(); // Solution gradient
   * ...
   *
   * const EnergyFunctor<decltype(soln_val), decltype(soln_grad), ...>
   * energy("e", "\\Psi", soln_val, soln_grad, ...);
   * </code>
   */
  template <typename... UnaryOpsSubSpaceFieldSolution>
  EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>
  energy_functor(
    const std::string &symbol_ascii,
    const std::string &symbol_latex,
    const UnaryOpsSubSpaceFieldSolution &... unary_op_field_solutions)
  {
    return EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>(
      symbol_ascii, symbol_latex, unary_op_field_solutions...);
  }


  template <typename ADNumberType,
            int dim,
            int spacedim = dim,
            typename... UnaryOpsSubSpaceFieldSolution,
            typename = typename std::enable_if<
              Differentiation::AD::is_ad_number<ADNumberType>::value>::type>
  WeakForms::Operators::UnaryOp<
    WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    WeakForms::Operators::UnaryOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>
  value(
    const WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...> &operand,
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<ADNumberType, dim, spacedim> &function)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>;
    using ScalarType =
      typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type;
    using OpType = UnaryOp<Op,
                           UnaryOpCodes::value,
                           ScalarType,
                           ADNumberType,
                           WeakForms::internal::DimPack<dim, spacedim>>;

    return OpType(operand, function);
  }



  // template <typename NumberType = double, int rank, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctor<rank, dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::TensorFunctor<rank, dim> &operand,
  //       const typename WeakForms::TensorFunctor<rank, dim>::
  //         template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = TensorFunctor<rank, dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function /*,NumberType()*/);
  // }



  // template <typename NumberType = double, int rank, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::SymmetricTensorFunctor<rank, dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::SymmetricTensorFunctor<rank, dim> &operand,
  //       const typename WeakForms::SymmetricTensorFunctor<rank, dim>::
  //         template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SymmetricTensorFunctor<rank, dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function);
  // }



  // template <typename NumberType = double, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::ScalarFunctionFunctor<dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::ScalarFunctionFunctor<dim> &operand,
  //       const typename WeakForms::ScalarFunctionFunctor<
  //         dim>::template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = ScalarFunctionFunctor<dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function);
  // }



  // template <typename NumberType = double, int rank, int dim>
  // WeakForms::Operators::UnaryOp<WeakForms::TensorFunctionFunctor<rank, dim>,
  //                               WeakForms::Operators::UnaryOpCodes::value,
  //                               NumberType>
  // value(const WeakForms::TensorFunctionFunctor<rank, dim> &operand,
  //       const typename WeakForms::TensorFunctionFunctor<rank, dim>::
  //         template function_type<NumberType> &function)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = TensorFunctionFunctor<rank, dim>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::value, NumberType>;

  //   return OpType(operand, function);
  // }
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



/* ==================== Class method definitions ==================== */

namespace WeakForms
{
  template <typename... UnaryOpsSubSpaceFieldSolution>
  template <typename NumberType, int dim, int spacedim>
  auto
  EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::operator()(
    const typename WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>::
      template function_type<NumberType, dim, spacedim> &function) const
  {
    return WeakForms::value<NumberType, dim, spacedim>(*this, function);
  }

} // namespace WeakForms



#ifndef DOXYGEN


namespace WeakForms
{
  // Decorator classes

  // template <int dim, int spacedim>
  // struct is_ad_functor<FieldSolution<dim, spacedim>> : std::true_type
  // {};

  // Unary operations

  template <typename ADNumberType,
            int dim,
            int spacedim,
            typename... UnaryOpsSubSpaceFieldSolution>
  struct is_ad_functor<Operators::UnaryOp<
    EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
    Operators::UnaryOpCodes::value,
    typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
    ADNumberType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_energy_functor_h
