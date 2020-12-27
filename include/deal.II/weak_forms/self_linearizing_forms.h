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

#ifndef dealii_weakforms_self_linearizing_forms_h
#define dealii_weakforms_self_linearizing_forms_h

#include <deal.II/base/config.h>

#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/unary_operators.h>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace SelfLinearization
  {
    namespace internal
    {
      template <typename FEValuesExtractors_t>
      struct SubSpaceExtractor;


      template <>
      struct SubSpaceExtractor<FEValuesExtractors::Scalar>
      {
        using type = WeakForms::SubSpaceExtractors::Scalar;
      };


      template <>
      struct SubSpaceExtractor<FEValuesExtractors::Vector>
      {
        using type = WeakForms::SubSpaceExtractors::Vector;
      };


      template <int rank>
      struct SubSpaceExtractor<FEValuesExtractors::Tensor<rank>>
      {
        using type = WeakForms::SubSpaceExtractors::Tensor<rank>;
      };


      template <int rank>
      struct SubSpaceExtractor<FEValuesExtractors::SymmetricTensor<rank>>
      {
        using type = WeakForms::SubSpaceExtractors::SymmetricTensor<rank>;
      };

      // Convert field solutions to a test function or trial solution.
      // This is required because we'll probe the arguments for the
      // functor supplied to the self-linearizing form, and from these
      // we'll have to construct the relevant residual (linear) and
      // linearization (bilinear) forms.
      struct ConvertTo
      {
        // For SubSpaceViews::Scalar and SubSpaceViews::Vector
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        test_function(
          const WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                              OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t = SubSpaceViewsType<SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TestFunction<dim, spacedim>;
          using Op      = SubSpaceViewsType<Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }

        // For SubSpaceViews::Scalar and SubSpaceViews::Vector
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        trial_solution(
          const WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                              OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t = SubSpaceViewsType<SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TrialSolution<dim, spacedim>;
          using Op      = SubSpaceViewsType<Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }


        // For SubSpaceViews::Tensor and SubSpaceViews::SymmetricTensor
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        test_function(const WeakForms::Operators::UnaryOp<
                      SubSpaceViewsType<rank, SpaceType>,
                      OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t =
            SubSpaceViewsType<rank, SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TestFunction<dim, spacedim>;
          using Op      = SubSpaceViewsType<rank, Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }

        // For SubSpaceViews::Tensor and SubSpaceViews::SymmetricTensor
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::UnaryOpCodes OpCode,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        trial_solution(const WeakForms::Operators::UnaryOp<
                       SubSpaceViewsType<rank, SpaceType>,
                       OpCode> &unary_op)
        {
          using SubSpaceViewFieldSolution_t =
            SubSpaceViewsType<rank, SpaceType>;
          using UnaryOp_t =
            WeakForms::Operators::UnaryOp<SubSpaceViewFieldSolution_t, OpCode>;
          constexpr unsigned int dim      = UnaryOp_t::dimension;
          constexpr unsigned int spacedim = UnaryOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TrialSolution<dim, spacedim>;
          using Op      = SubSpaceViewsType<rank, Space_t>;
          using OpType  = WeakForms::Operators::UnaryOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = unary_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }
      };


      // Cartesian product of variadic template types
      // Adapted from https://stackoverflow.com/a/9145665
      namespace TemplateOuterProduct
      {
        template <typename... Ts>
        struct TypeList
        {};
        template <typename T1, typename T2>
        struct TypePair
        {};

        // Concatenation
        template <typename... T>
        struct Concatenate;
        template <typename... Ts, typename... Us>
        struct Concatenate<TypeList<Ts...>, TypeList<Us...>>
        {
          using type = TypeList<Ts..., Us...>;
        };

        // Outer Product
        template <typename T, typename U>
        struct OuterProduct;

        // Partially specialise the empty case for the first TypeList.
        template <typename... Us>
        struct OuterProduct<TypeList<>, TypeList<Us...>>
        {
          using type = TypeList<>;
        };

        // The general case for two TypeLists. Process:
        // 1. Expand out the head of the first TypeList with the full second
        // TypeList.
        // 2. Recurse the tail of the first TypeList.
        // 3. Concatenate the two TypeLists.
        template <typename T, typename... Ts, typename... Us>
        struct OuterProduct<TypeList<T, Ts...>, TypeList<Us...>>
        {
          using type = typename Concatenate<
            TypeList<TypePair<T, Us>...>,
            typename OuterProduct<TypeList<Ts...>,
                                  TypeList<Us...>>::type>::type;
        };
      } // namespace TemplateOuterProduct


      // Ensure that template arguments contain no duplicates.
      // Adapted from https://stackoverflow.com/a/34122593
      namespace TemplateRestrictions
      {
        template <typename T, typename... List>
        struct IsContained;

        template <typename T, typename Head, typename... Tail>
        struct IsContained<T, Head, Tail...>
        {
          enum
          {
            value =
              std::is_same<T, Head>::value || IsContained<T, Tail...>::value
          };
        };

        template <typename T>
        struct IsContained<T>
        {
          enum
          {
            value = false
          };
        };

        template <typename... List>
        struct IsUnique;

        template <typename Head, typename... Tail>
        struct IsUnique<Head, Tail...>
        {
          enum
          {
            value =
              !IsContained<Head, Tail...>::value && IsUnique<Tail...>::value
          };
        };

        template <>
        struct IsUnique<>
        {
          enum
          {
            value = true
          };
        };

        template <typename... Ts>
        struct NoDuplicates
        {
          static_assert(IsUnique<Ts...>::value, "No duplicate types allowed.");
        };
      } // namespace TemplateRestrictions

    } // namespace internal


    /**
     * OP: (AutoDifferentiableFunctor)
     *
     * First derivatives of this form produce a ResidualForm.
     */
    template <typename ADFunctor, typename... FieldArgs>
    class EnergyFunctional
    {};

    /**
     * OP: (Variation, SymbolicFunctor)
     *
     * This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    class ResidualForm
    {};

    /**
     * OP: (Variation, SymbolicFunctor, Linearization)
     *
     * This class gets converted into a LinearForm.
     * First derivatives of this form produce a BilinearForm through the
     * LinearizationForm
     */
    class LinearizationForm
    {
    private:
      // friend EnergyFunctional;
      // friend ResidualForm;
      LinearizationForm() = default;
    };
  } // namespace SelfLinearization

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



// namespace WeakForms
// {
//   // template <typename TestSpaceOp, typename TrialSpaceOp>
//   // BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>
//   // bilinear_form(const TestSpaceOp & test_space_op,
//   //               const TrialSpaceOp &trial_space_op)
//   // {
//   //   return BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>(test_space_op,
//   //                                                        trial_space_op);
//   // }

//   template <typename ADFunctor, typename... FieldArgs>
//   SelfLinearization::EnergyFunctional<ADFunctor, FieldArgs...>
//   ad_energy_functional_form(const ADFunctor &functor_op,
//                             const FieldArgs &... dependent_fields)
//   {
//     return SelfLinearization::EnergyFunctional<ADFunctor, FieldArgs...>(
//       functor_op, dependent_fields...);
//   }

// } // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_self_linearizing_forms_h
