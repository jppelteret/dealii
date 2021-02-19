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

// #include <boost/core/demangle.hpp>

// TODO: Are all of these needed?
#include <deal.II/weak_forms/assembler.h>
#include <deal.II/weak_forms/bilinear_forms.h>
#include <deal.II/weak_forms/differentiation.h>
#include <deal.II/weak_forms/energy_functor.h>
// #include <deal.II/weak_forms/functors.h> // Needed?
#include <deal.II/weak_forms/integral.h>
#include <deal.II/weak_forms/linear_forms.h>
#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/subspace_views.h>
#include <deal.II/weak_forms/symbolic_operators.h>
#include <deal.II/weak_forms/type_traits.h>

#include <string>
#include <type_traits>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace SelfLinearization
  {
    namespace internal
    {
      // Make the link between FEValuesExtractors and the weak form
      // SubSpaceExtractors
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
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  std::size_t                                solution_index,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        test_function(
          const WeakForms::Operators::SymbolicOp<
            SubSpaceViewsType<SpaceType>,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>> &symbolic_op)
        {
          using SubSpaceViewFieldSolution_t = SubSpaceViewsType<SpaceType>;
          using UnaryFieldOp_t              = WeakForms::Operators::SymbolicOp<
            SubSpaceViewFieldSolution_t,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>>;
          constexpr unsigned int dim      = UnaryFieldOp_t::dimension;
          constexpr unsigned int spacedim = UnaryFieldOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TestFunction<dim, spacedim>;
          using Op      = SubSpaceViewsType<Space_t>;
          using OpType  = WeakForms::Operators::SymbolicOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = symbolic_op.get_operand();
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
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  std::size_t                                solution_index,
                  typename = typename std::enable_if<
                    is_field_solution<SpaceType>::value>::type>
        static auto
        trial_solution(
          const WeakForms::Operators::SymbolicOp<
            SubSpaceViewsType<SpaceType>,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>> &symbolic_op)
        {
          using SubSpaceViewFieldSolution_t = SubSpaceViewsType<SpaceType>;
          using UnaryFieldOp_t              = WeakForms::Operators::SymbolicOp<
            SubSpaceViewFieldSolution_t,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>>;
          constexpr unsigned int dim      = UnaryFieldOp_t::dimension;
          constexpr unsigned int spacedim = UnaryFieldOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TrialSolution<dim, spacedim>;
          using Op      = SubSpaceViewsType<Space_t>;
          using OpType  = WeakForms::Operators::SymbolicOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = symbolic_op.get_operand();
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
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  std::size_t                                solution_index,
                  typename = typename std::enable_if<
                    is_field_solution_op<SpaceType>::value>::type>
        static auto
        test_function(
          const WeakForms::Operators::SymbolicOp<
            SubSpaceViewsType<rank, SpaceType>,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>> &symbolic_op)
        {
          using SubSpaceViewFieldSolution_t =
            SubSpaceViewsType<rank, SpaceType>;
          using UnaryFieldOp_t = WeakForms::Operators::SymbolicOp<
            SubSpaceViewFieldSolution_t,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>>;
          constexpr unsigned int dim      = UnaryFieldOp_t::dimension;
          constexpr unsigned int spacedim = UnaryFieldOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TestFunction<dim, spacedim>;
          using Op      = SubSpaceViewsType<rank, Space_t>;
          using OpType  = WeakForms::Operators::SymbolicOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = symbolic_op.get_operand();
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
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  std::size_t                                solution_index,
                  typename = typename std::enable_if<
                    is_field_solution_op<SpaceType>::value>::type>
        static auto
        trial_solution(
          const WeakForms::Operators::SymbolicOp<
            SubSpaceViewsType<rank, SpaceType>,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>> &symbolic_op)
        {
          using SubSpaceViewFieldSolution_t =
            SubSpaceViewsType<rank, SpaceType>;
          using UnaryFieldOp_t = WeakForms::Operators::SymbolicOp<
            SubSpaceViewFieldSolution_t,
            OpCode,
            void,
            WeakForms::internal::SolutionIndex<solution_index>>;
          constexpr unsigned int dim      = UnaryFieldOp_t::dimension;
          constexpr unsigned int spacedim = UnaryFieldOp_t::space_dimension;

          using namespace WeakForms;

          using Space_t = TrialSolution<dim, spacedim>;
          using Op      = SubSpaceViewsType<rank, Space_t>;
          using OpType  = WeakForms::Operators::SymbolicOp<Op, OpCode>;
          using FEValuesExtractor_t =
            typename SubSpaceViewFieldSolution_t::FEValuesExtractorType;
          using SubSpaceExtractor_t =
            typename SubSpaceExtractor<FEValuesExtractor_t>::type;

          // Rebuild the subspace extractor from that used to produce the field
          // solution view
          const auto &field_solution_ss_op = symbolic_op.get_operand();
          const SubSpaceExtractor_t extractor(
            field_solution_ss_op.get_extractor(),
            field_solution_ss_op.get_space().get_field_ascii_raw(),
            field_solution_ss_op.get_space().get_field_latex_raw());
          // And now apply it to the (sub)space that we wish convert to
          const Space_t space;
          const Op      operand(space[extractor]);
          return OpType(operand);
        }


        // Each @p SymbolicOpSubSpaceFieldSolution is expected to be a
        // Operators::SymbolicOp<SubSpaceViews::[Scalar/Vector/Tensor/SymmetricTensor]>>
        // Since we can't convert the underlying SubSpaceViewsType (its a fixed
        // FieldSolution) we just ask for what the expected return values of the
        // above helper functions would be.
        template <typename SymbolicOpSubSpaceFieldSolution>
        using test_function_t = decltype(
          test_function(std::declval<SymbolicOpSubSpaceFieldSolution>()));

        template <typename SymbolicOpSubSpaceFieldSolution>
        using trial_solution_t = decltype(
          trial_solution(std::declval<SymbolicOpSubSpaceFieldSolution>()));
      };


      // Ensure that template arguments contain no duplicates.
      // Adapted from https://stackoverflow.com/a/34122593
      namespace TemplateRestrictions
      {
        template <typename T, typename... List>
        struct IsContained;


        template <typename T, typename Head, typename... Tail>
        struct IsContained<T, Head, Tail...>
        {
          static constexpr bool value =
            std::is_same<T, Head>::value || IsContained<T, Tail...>::value;
        };


        template <typename T>
        struct IsContained<T>
        {
          static constexpr bool value = false;
        };


        template <typename... List>
        struct IsUnique;


        template <typename Head, typename... Tail>
        struct IsUnique<Head, Tail...>
        {
          static constexpr bool value =
            !IsContained<Head, Tail...>::value && IsUnique<Tail...>::value;
        };


        template <>
        struct IsUnique<>
        {
          static constexpr bool value = true;
        };


        template <typename... Ts>
        struct EnforceNoDuplicates
        {
          static_assert(IsUnique<Ts...>::value, "No duplicate types allowed.");

          static constexpr bool value = IsUnique<Ts...>::value;
        };


        template <typename T, typename... Us>
        struct is_subspace_field_solution_op
        {
          static constexpr bool value =
            is_subspace_field_solution_op<T>::value &&
            is_subspace_field_solution_op<Us...>::value;
        };

        // Scalar and Vector subspaces
        template <template <class> typename SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  std::size_t                                solution_index>
        struct is_subspace_field_solution_op<WeakForms::Operators::SymbolicOp<
          SubSpaceViewsType<SpaceType>,
          OpCode,
          void,
          WeakForms::internal::SolutionIndex<solution_index>>>
        {
          static constexpr bool value =
            is_field_solution<SubSpaceViewsType<SpaceType>>::value &&
            is_subspace_view<SubSpaceViewsType<SpaceType>>::value;
        };

        // Tensor and SymmetricTensor subspaces
        template <template <int, class> typename SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  std::size_t                                solution_index>
        struct is_subspace_field_solution_op<WeakForms::Operators::SymbolicOp<
          SubSpaceViewsType<rank, SpaceType>,
          OpCode,
          void,
          WeakForms::internal::SolutionIndex<solution_index>>>
        {
          static constexpr bool value =
            is_field_solution<SubSpaceViewsType<rank, SpaceType>>::value &&
            is_subspace_view<SubSpaceViewsType<rank, SpaceType>>::value;
        };

        template <typename T>
        struct is_subspace_field_solution_op<T> : std::false_type
        {};

        template <typename... FieldArgs>
        struct EnforceIsSymbolicOpSubspaceFieldSolution
        {
          static_assert(
            is_subspace_field_solution_op<FieldArgs...>::value,
            "Template arguments must be unary operation subspace field solutions. "
            "You might have used a test function or trial solution, or perhaps "
            "have not used a sub-space extractor.");

          static constexpr bool value = true;
        };
      } // namespace TemplateRestrictions
    }   // namespace internal


    /**
     * A special form that consumes an energy functor and produces
     * both the associated linear form and consistently-linearized
     * bilinear form associated with the energy functional.
     *
     * The @p EnergyFunctional form is supplied with the finite element fields upon
     * which the @p Functor is parameterized. It then self-linearizes the discrete
     * problem (i.e. at the finite element level) to produce the linear and
     * bilinear forms. However, this class doesn't directly know how the energy
     * functor itself is linearized. So the derivatives of the energy functor
     * with respect to the various field parameters are given by the energy
     * functor itself. We employ automatic or symbolic differentiation to
     * perform that task. The local description of the energy (i.e. at the
     * quadrature point level)
     * is given by the @p Functor.
     *
     * This is fair trade between the convenience of compile-time
     * expansions for the derivatives of the energy functional, and some
     * run-time derivatives of the (potentially complex) constitutive
     * laws that the @p Functor describes. The functor is only evaluated
     * at quadrature points, so the computational cost associated with
     * the calculation of those derivatives is kept to a minimum.
     * It also means that we can take care of most of the bookkeeping and
     * implementational details surrounding AD and SD. The user then needs a
     * "minimal" understanding of how these parts of the framework work in order
     * to use this feature.
     */
    template <typename Functor, typename... SymbolicOpsSubSpaceFieldSolution>
    class EnergyFunctional
    {
      static_assert(
        is_ad_functor_op<Functor>::value || is_sd_functor_op<Functor>::value,
        "The SelfLinearizing::EnergyFunctional class is designed to work with AD or SD functors.");

      // All template parameter types must be unary operators
      // for subspaces of a field solution.
      static_assert(
        internal::TemplateRestrictions::
          EnforceIsSymbolicOpSubspaceFieldSolution<
            SymbolicOpsSubSpaceFieldSolution...>::value,
        "Template arguments must be unary operation subspace field solutions. "
        "You might have used a test function or trial solution, or perhaps "
        "have not used a sub-space extractor.");

      // We cannot permit multiple instance of the same unary operations
      // as a part of the template parameter pack. This would imply that
      // we want the user to define a functor that takes in multiple instances
      // of the same field variable, which does not make sense.
      // static_assert(internal::TemplateRestrictions::EnforceNoDuplicates<
      //                 SymbolicOpsSubSpaceFieldSolution...>::value,
      //               "No duplicate types allowed.");

      // static_assert(
      //   is_symbolic_op<Functor>::value,
      //   "The SelfLinearizing::EnergyFunctional class is designed to work a
      //   unary operation as a functor.");

    public:
      EnergyFunctional(
        const Functor &functor_op,
        const SymbolicOpsSubSpaceFieldSolution &... symbolic_op_field_solutions)
        : functor_op(functor_op)
        , symbolic_op_field_solutions(symbolic_op_field_solutions...)
      {}

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return "SelfLinearizingEnergyFunctional(" +
               functor_op.as_ascii(decorator) + ")";
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        // const std::string lbrace = Utilities::LaTeX::l_square_brace;
        // const std::string rbrace = Utilities::LaTeX::r_square_brace;

        // constexpr unsigned int n_contracting_indices_tf =
        //   WeakForms::Utilities::IndexContraction<TestSpaceOp,
        //                                          Functor>::n_contracting_indices;
        // const std::string symb_mult_tf =
        //   Utilities::LaTeX::get_symbol_multiply(n_contracting_indices_tf);

        // return lbrace + test_space_op.as_latex(decorator) + symb_mult_tf +
        //        functor_op.as_latex(decorator) + rbrace;

        return "SelfLinearizingEnergyFunctional(" +
               functor_op.as_latex(decorator) + ")";
      }

      // ===== Section: Construct assembly operation =====

      UpdateFlags
      get_update_flags() const
      {
        return unpack_update_flags(get_field_args());
      }

      const Functor &
      get_functor() const
      {
        return functor_op;
      }

      const std::tuple<SymbolicOpsSubSpaceFieldSolution...> &
      get_field_args() const
      {
        return symbolic_op_field_solutions;
      }

      // ===== Section: Integration =====

      auto
      dV() const
      {
        return integrate(*this, VolumeIntegral());
      }

      auto
      dV(const typename VolumeIntegral::subdomain_t subdomain) const
      {
        return dV(std::set<typename VolumeIntegral::subdomain_t>{subdomain});
      }

      auto
      dV(const std::set<typename VolumeIntegral::subdomain_t> &subdomains) const
      {
        return integrate(*this, VolumeIntegral(subdomains));
      }

      auto
      dA() const
      {
        return integrate(*this, BoundaryIntegral());
      }

      auto
      dA(const typename BoundaryIntegral::subdomain_t boundary) const
      {
        return dA(std::set<typename BoundaryIntegral::subdomain_t>{boundary});
      }

      auto
      dA(const std::set<typename BoundaryIntegral::subdomain_t> &boundaries)
        const
      {
        return integrate(*this, BoundaryIntegral(boundaries));
      }

      auto
      dI() const
      {
        return integrate(*this, InterfaceIntegral());
      }

      auto
      dI(const typename InterfaceIntegral::subdomain_t interface) const
      {
        return dI(std::set<typename InterfaceIntegral::subdomain_t>{interface});
      }

      auto
      dI(const std::set<typename InterfaceIntegral::subdomain_t> &interfaces)
        const
      {
        return integrate(*this, InterfaceIntegral(interfaces));
      }

    private:
      const Functor functor_op;
      const std::tuple<SymbolicOpsSubSpaceFieldSolution...>
        symbolic_op_field_solutions;

      // =============
      // AD operations
      // =============

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex,
                typename SymbolicOpField,
                typename T = Functor>
      auto
      get_functor_first_derivative(
        const SymbolicOpField &field,
        typename std::enable_if<is_ad_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = SymbolicOpField::dimension;
        constexpr int spacedim = SymbolicOpField::space_dimension;

        using FunctorScalar_t = typename Functor::scalar_type;
        using FieldValue_t =
          typename SymbolicOpField::template value_type<AssemblerScalar_t>;
        using DiffOpResult_t =
          WeakForms::internal::Differentiation::DiffOpResult<FunctorScalar_t,
                                                             FieldValue_t>;

        using DiffOpValue_t = typename DiffOpResult_t::type;
        using DiffOpFunction_t =
          typename DiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For AD types, the derivative_extractor will be a FEValues::Extractor.
        const Functor &functor = this->get_functor();
        const auto &   derivative_extractor =
          functor.template get_derivative_extractor<FieldIndex>(field);

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return DiffOpResult_t::template get_functor<dim, spacedim>(
          "Df",
          "D(f)",
          [functor, derivative_extractor](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<std::string> &        solution_names) {
            // We need to fetch the helper from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &helper = functor.get_ad_helper(scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<Vector<FunctorScalar_t>> &gradients =
              functor.get_gradients(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                helper.extract_gradient_component(gradients[q_point],
                                                  derivative_extractor));

            return out;
          },
          functor,
          field);
      }

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex_1,
                std::size_t FieldIndex_2,
                typename SymbolicOpField_1,
                typename SymbolicOpField_2,
                typename T = Functor>
      auto
      get_functor_second_derivative(
        const SymbolicOpField_1 &field_1,
        const SymbolicOpField_2 &field_2,
        typename std::enable_if<is_ad_functor_op<T>::value>::type * =
          nullptr) const
      {
        static_assert(SymbolicOpField_1::dimension ==
                        SymbolicOpField_2::dimension,
                      "Dimension mismatch");
        static_assert(SymbolicOpField_1::space_dimension ==
                        SymbolicOpField_2::space_dimension,
                      "Space dimension mismatch");

        constexpr int dim      = SymbolicOpField_1::dimension;
        constexpr int spacedim = SymbolicOpField_1::space_dimension;

        using FunctorScalar_t = typename Functor::scalar_type;
        using FieldValue_1_t =
          typename SymbolicOpField_1::template value_type<AssemblerScalar_t>;
        using FieldValue_2_t =
          typename SymbolicOpField_2::template value_type<AssemblerScalar_t>;
        using FirstDiffOpResult_t =
          WeakForms::internal::Differentiation::DiffOpResult<FunctorScalar_t,
                                                             FieldValue_1_t>;
        using SecondDiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<typename FirstDiffOpResult_t::type, FieldValue_2_t>;

        using DiffOpValue_t = typename SecondDiffOpResult_t::type;
        using DiffOpFunction_t =
          typename SecondDiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        const Functor &functor = this->get_functor();
        const auto &   derivative_1_extractor =
          functor.template get_derivative_extractor<FieldIndex_1>(field_1);
        const auto &derivative_2_extractor =
          functor.template get_derivative_extractor<FieldIndex_2>(field_2);

        // The functor may only be temporary, so pass it in as a copy.
        // The extractors are specific to this operation, so they definitely
        // must be passed by copy.
        return SecondDiffOpResult_t::template get_functor<dim, spacedim>(
          "D2f",
          "D^{2}(f)",
          [functor, derivative_1_extractor, derivative_2_extractor](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<std::string> &        solution_names) {
            // We need to fetch the helper from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &helper = functor.get_ad_helper(scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<FullMatrix<FunctorScalar_t>> &hessians =
              functor.get_hessians(scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                helper.extract_hessian_component(hessians[q_point],
                                                 derivative_1_extractor,
                                                 derivative_2_extractor));

            return out;
          },
          functor,
          field_1,
          field_2);
      }

      // =============
      // SD operations
      // =============

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex,
                typename SymbolicOpField,
                typename T = Functor>
      auto
      get_functor_first_derivative(
        const SymbolicOpField &field,
        typename std::enable_if<is_sd_functor_op<T>::value>::type * =
          nullptr) const
      {
        constexpr int dim      = SymbolicOpField::dimension;
        constexpr int spacedim = SymbolicOpField::space_dimension;

        // SD expressions can represent anything, so it doesn't make sense to
        // ask the functor for this type. We expect the result to be castable
        // into the Assembler's scalar type.
        using FunctorScalar_t = AssemblerScalar_t;
        using FieldValue_t =
          typename SymbolicOpField::template value_type<AssemblerScalar_t>;
        using DiffOpResult_t =
          WeakForms::internal::Differentiation::DiffOpResult<FunctorScalar_t,
                                                             FieldValue_t>;

        using DiffOpValue_t = typename DiffOpResult_t::type;
        using DiffOpFunction_t =
          typename DiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        // For SD types, the derivative_extractor an SD::Expression or tensor of
        // expressions that correspond to the solution field that is being
        // derived with respect to.
        const Functor &functor = this->get_functor();
        const auto &   first_derivative =
          functor.template get_symbolic_first_derivative<FieldIndex>();

        // The functor may only be temporary, so pass it in as a copy.
        // The extractor is specific to this operation, so it definitely
        // must be passed by copy.
        return DiffOpResult_t::template get_functor<dim, spacedim>(
          "Df",
          "D(f)",
          [functor, first_derivative](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<std::string> &        solution_names) {
            // We need to fetch the optimizer from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &optimizer =
              functor.template get_batch_optimizer<FunctorScalar_t>(
                scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<std::vector<FunctorScalar_t>>
              &evaluated_dependent_functions =
                functor
                  .template get_evaluated_dependent_functions<FunctorScalar_t>(
                    scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            // Note: We should not use the evaluated variables that are stored
            // in the optimizer itself. They will only store the values computed
            // at the last call to optimizer.substitute(), which should be the
            // values at the last evaluated quadrature point.
            // We rather follow the same approach as for AD, and store the
            // evaluated variables elsewhere until we want to evaluate them
            // with some centralized optimizer.
            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                optimizer.extract(first_derivative,
                                  evaluated_dependent_functions[q_point]));

            return out;
          },
          functor,
          field);
      }

      template <typename AssemblerScalar_t,
                std::size_t FieldIndex_1,
                std::size_t FieldIndex_2,
                typename SymbolicOpField_1,
                typename SymbolicOpField_2,
                typename T = Functor>
      auto
      get_functor_second_derivative(
        const SymbolicOpField_1 &field_1,
        const SymbolicOpField_2 &field_2,
        typename std::enable_if<is_sd_functor_op<T>::value>::type * =
          nullptr) const
      {
        static_assert(SymbolicOpField_1::dimension ==
                        SymbolicOpField_2::dimension,
                      "Dimension mismatch");
        static_assert(SymbolicOpField_1::space_dimension ==
                        SymbolicOpField_2::space_dimension,
                      "Space dimension mismatch");

        constexpr int dim      = SymbolicOpField_1::dimension;
        constexpr int spacedim = SymbolicOpField_1::space_dimension;

        // SD expressions can represent anything, so it doesn't make sense to
        // ask the functor for this type. We expect the result to be castable
        // into the Assembler's scalar type.
        using FunctorScalar_t = AssemblerScalar_t;
        using FieldValue_1_t =
          typename SymbolicOpField_1::template value_type<AssemblerScalar_t>;
        using FieldValue_2_t =
          typename SymbolicOpField_2::template value_type<AssemblerScalar_t>;
        using FirstDiffOpResult_t =
          WeakForms::internal::Differentiation::DiffOpResult<FunctorScalar_t,
                                                             FieldValue_1_t>;
        using SecondDiffOpResult_t = WeakForms::internal::Differentiation::
          DiffOpResult<typename FirstDiffOpResult_t::type, FieldValue_2_t>;

        using DiffOpValue_t = typename SecondDiffOpResult_t::type;
        using DiffOpFunction_t =
          typename SecondDiffOpResult_t::template function_type<dim>;

        static_assert(
          std::is_same<std::vector<DiffOpValue_t>,
                       typename DiffOpFunction_t::result_type>::value,
          "Expected same result type.");

        const Functor &functor = this->get_functor();
        const auto &   second_derivative =
          functor.template get_symbolic_second_derivative<FieldIndex_1,
                                                          FieldIndex_2>();

        // The functor may only be temporary, so pass it in as a copy.
        // The extractors are specific to this operation, so they definitely
        // must be passed by copy.
        return SecondDiffOpResult_t::template get_functor<dim, spacedim>(
          "D2f",
          "D^{2}(f)",
          [functor, second_derivative](
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<std::string> &        solution_names) {
            // We need to fetch the optimizer from Scratch (rather than passing
            // it into this lambda function) to avoid working with the same copy
            // of this object on multiple threads.
            const auto &optimizer =
              functor.template get_batch_optimizer<FunctorScalar_t>(
                scratch_data);
            // The return result from the differentiation is also not shared
            // between threads. But we can reuse the same object many times
            // since its stored in Scratch.
            const std::vector<std::vector<FunctorScalar_t>>
              &evaluated_dependent_functions =
                functor
                  .template get_evaluated_dependent_functions<FunctorScalar_t>(
                    scratch_data);

            std::vector<DiffOpValue_t>         out;
            const FEValuesBase<dim, spacedim> &fe_values =
              scratch_data.get_current_fe_values();
            out.reserve(fe_values.n_quadrature_points);

            // Note: We should not use the evaluated variables that are stored
            // in the optimizer itself. They will only store the values computed
            // at the last call to optimizer.substitute(), which should be the
            // values at the last evaluated quadrature point.
            // We rather follow the same approach as for AD, and store the
            // evaluated variables elsewhere until we want to evaluate them
            // with some centralized optimizer.
            for (const auto &q_point : fe_values.quadrature_point_indices())
              out.emplace_back(
                optimizer.extract(second_derivative,
                                  evaluated_dependent_functions[q_point]));

            return out;
          },
          functor,
          field_1,
          field_2);
      }

      // =============================
      // Self-linearization operations
      // =============================

      // Provide access to accumulation function
      template <int dim2,
                int spacedim,
                typename ScalarType,
                bool use_vectorization>
      friend class WeakForms::AssemblerBase;

      template <enum WeakForms::internal::AccumulationSign OpSign,
                typename AssemblerType,
                typename IntegralType>
      void
      accumulate_into(AssemblerType &     assembler,
                      const IntegralType &integral_operation) const
      {
        unpack_accumulate_linear_form_into<OpSign>(assembler,
                                                   integral_operation,
                                                   get_field_args());

        unpack_accumulate_bilinear_form_into<OpSign>(assembler,
                                                     integral_operation,
                                                     get_field_args(),
                                                     get_field_args());
      }

      // === Recursive function ===
      // All patterns constructed below follow the approach
      // laid out here:
      // https://stackoverflow.com/a/6894436

      // Get update flags from a unary op
      template <std::size_t I = 0, typename... SymbolicOpType>
        inline typename std::enable_if <
        I<sizeof...(SymbolicOpType), UpdateFlags>::type
        unpack_update_flags(const std::tuple<SymbolicOpType...>
                              &symbolic_op_field_solutions) const
      {
        return std::get<I>(symbolic_op_field_solutions).get_update_flags() |
               unpack_update_flags<I + 1, SymbolicOpType...>(
                 symbolic_op_field_solutions);
      }

      // Get update flags from a unary op: End point
      template <std::size_t I = 0, typename... SymbolicOpType>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType),
                                     UpdateFlags>::type
      unpack_update_flags(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
      {
        // Do nothing
        return UpdateFlags::update_default;
      }

      // Create linear forms
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType>
        inline typename std::enable_if <
        I<sizeof...(SymbolicOpType), void>::type
        unpack_accumulate_linear_form_into(
          AssemblerType &                      assembler,
          const IntegralType &                 integral_operation,
          const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions)
          const
      {
        using AssemblerScalar_t = typename AssemblerType::scalar_type;

        const auto &field_solution = std::get<I>(symbolic_op_field_solutions);
        const auto  test_function =
          internal::ConvertTo::test_function(field_solution);

        const auto linear_form = WeakForms::linear_form(
          test_function,
          get_functor_first_derivative<AssemblerScalar_t, I>(field_solution));
        const auto integrated_linear_form =
          WeakForms::value(integral_operation, linear_form);

        if (OpSign == WeakForms::internal::AccumulationSign::plus)
          {
            assembler += integrated_linear_form;
          }
        else
          {
            Assert(OpSign == WeakForms::internal::AccumulationSign::minus,
                   ExcInternalError());
            assembler -= integrated_linear_form;
          }

        // Move on to the next form:
        // This effectively traverses the list of dependent fields, creating the
        // linear forms associated with the residual starting from first to
        // last.
        unpack_accumulate_linear_form_into<OpSign, I + 1>(
          assembler, integral_operation, symbolic_op_field_solutions);
      }

      // Create linear forms: End point
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
      unpack_accumulate_linear_form_into(
        AssemblerType &                      assembler,
        const IntegralType &                 integral_operation,
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
      {
        // Do nothing
      }

      // Create bilinear forms
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                std::size_t                                J = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType_1,
                typename... SymbolicOpType_2>
          inline typename std::enable_if < I < sizeof...(SymbolicOpType_1) &&
        J<sizeof...(SymbolicOpType_2), void>::type
        unpack_accumulate_bilinear_form_into(
          AssemblerType &                        assembler,
          const IntegralType &                   integral_operation,
          const std::tuple<SymbolicOpType_1...> &symbolic_op_field_solutions_1,
          const std::tuple<SymbolicOpType_2...> &symbolic_op_field_solutions_2)
          const
      {
        using AssemblerScalar_t = typename AssemblerType::scalar_type;

        const auto &field_solution_1 =
          std::get<I>(symbolic_op_field_solutions_1);
        const auto &field_solution_2 =
          std::get<J>(symbolic_op_field_solutions_2);
        const auto test_function =
          internal::ConvertTo::test_function(field_solution_1);
        const auto trial_solution =
          internal::ConvertTo::trial_solution(field_solution_2);

        const auto bilinear_form = WeakForms::bilinear_form(
          test_function,
          get_functor_second_derivative<AssemblerScalar_t, I, J>(
            field_solution_1, field_solution_2),
          trial_solution);
        const auto integrated_bilinear_form =
          WeakForms::value(integral_operation, bilinear_form);

        if (OpSign == WeakForms::internal::AccumulationSign::plus)
          {
            assembler += integrated_bilinear_form;
          }
        else
          {
            Assert(OpSign == WeakForms::internal::AccumulationSign::minus,
                   ExcInternalError());
            assembler -= integrated_bilinear_form;
          }

        // Move on to the next forms:
        // This effectively traverses the list of dependent fields, generating
        // the bilinear forms associated with the linearization.
        //
        // Step 1: Linearize this linear form with respect to all field
        // variables. This basically traverses the row subblock of the system
        // and produces all column subblocks.
        unpack_accumulate_bilinear_form_into<OpSign, I, J + 1>(
          assembler,
          integral_operation,
          symbolic_op_field_solutions,
          symbolic_op_field_solutions);
        // Step 2: Only move on to the next row if we're at the zeroth column.
        // This is because the above operation traverses all columns in a row.
        if (J == 0)
          unpack_accumulate_bilinear_form_into<OpSign, I + 1, J>(
            assembler,
            integral_operation,
            symbolic_op_field_solutions,
            symbolic_op_field_solutions);
      }

      // Create bilinear forms: End point
      template <enum WeakForms::internal::AccumulationSign OpSign,
                std::size_t                                I = 0,
                std::size_t                                J = 0,
                typename AssemblerType,
                typename IntegralType,
                typename... SymbolicOpType_1,
                typename... SymbolicOpType_2>
      inline typename std::enable_if<I == sizeof...(SymbolicOpType_1) ||
                                       J == sizeof...(SymbolicOpType_2),
                                     void>::type
      unpack_accumulate_bilinear_form_into(
        AssemblerType &                        assembler,
        const IntegralType &                   integral_operation,
        const std::tuple<SymbolicOpType_1...> &symbolic_op_field_solutions_1,
        const std::tuple<SymbolicOpType_2...> &symbolic_op_field_solutions_2)
        const
      {
        // Do nothing
      }
    }; // class EnergyFunctional



    /**
     * TODO: Implement this
     */
    template <typename... SymbolicOpsSubSpaceFieldSolution>
    class ResidualForm
    {};
  } // namespace SelfLinearization

} // namespace WeakForms



/* ======================== Convenience functions ======================== */


namespace WeakForms
{
  template <typename Functor, typename... FieldArgs>
  SelfLinearization::EnergyFunctional<Functor, FieldArgs...>
  energy_functional_form(const Functor &functor_op,
                         const FieldArgs &... dependent_fields)
  {
    return SelfLinearization::EnergyFunctional<Functor, FieldArgs...>(
      functor_op, dependent_fields...);
  }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename... SymbolicOpsSubSpaceFieldSolution>
  struct is_self_linearizing_form<
    SelfLinearization::EnergyFunctional<SymbolicOpsSubSpaceFieldSolution...>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_self_linearizing_forms_h
