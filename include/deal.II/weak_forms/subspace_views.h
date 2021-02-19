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

#ifndef dealii_weakforms_subspace_views_h
#define dealii_weakforms_subspace_views_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/weak_forms/solution_storage.h>
#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/symbolic_operators.h>
#include <deal.II/weak_forms/type_traits.h>


DEAL_II_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  namespace SelfLinearization
  {
    namespace internal
    {
      struct ConvertTo;
    } // namespace internal
  }   // namespace SelfLinearization



  /* ----- Finite element subspaces: Test functions and trial solutions ----- */


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const SubSpaceViewsType<SpaceType> &operand);


  template <template <int, class> typename SubSpaceViewsType,
            int rank,
            typename SpaceType>
  WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank, SpaceType>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const SubSpaceViewsType<rank, SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const SubSpaceViewsType<SpaceType> &operand);


  template <template <int, class> typename SubSpaceViewsType,
            int rank,
            typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const SubSpaceViewsType<rank, SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
  symmetric_gradient(const SubSpaceViewsType<SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::divergence>
  divergence(const SubSpaceViewsType<SpaceType> &operand);


  template <template <int, class> typename SubSpaceViewsType,
            int rank,
            typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::divergence>
  divergence(const SubSpaceViewsType<rank, SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
                                   WeakForms::Operators::SymbolicOpCodes::curl>
  curl(const SubSpaceViewsType<SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::laplacian>
  laplacian(const SubSpaceViewsType<SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::hessian>
  hessian(const SubSpaceViewsType<SpaceType> &operand);


  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative>
  third_derivative(const SubSpaceViewsType<SpaceType> &operand);



  /* ------------- Finite element subspaces: Field solutions ------------- */


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <int, class> typename SubSpaceViewsType,
            int rank,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <int, class> typename SubSpaceViewsType,
            int rank,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(
    const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::symmetric_gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  symmetric_gradient(
    const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::divergence,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  divergence(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <int, class> typename SubSpaceViewsType,
            int rank,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::divergence,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  divergence(
    const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::curl,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  curl(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::laplacian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  laplacian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::hessian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  hessian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);


  template <std::size_t solution_index = 0,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  third_derivative(
    const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand);

} // namespace WeakForms

#endif // DOXYGEN



namespace WeakForms
{
  namespace SubSpaceViews
  {
    template <typename SpaceType, typename ExtractorType>
    class SubSpaceViewBase
    {
    public:
      using FEValuesExtractorType = ExtractorType;

      virtual ~SubSpaceViewBase() = default;

      virtual SubSpaceViewBase *
      clone() const = 0;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return space.as_ascii(decorator);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        return space.as_latex(decorator);
      }

      std::string
      get_field_ascii(const SymbolicDecorations &decorator) const
      {
        return space.get_field_ascii(decorator);
      }

      std::string
      get_field_latex(const SymbolicDecorations &decorator) const
      {
        return space.get_field_latex(decorator);
      }

      std::string
      get_symbol_ascii(const SymbolicDecorations &decorator) const
      {
        return space.get_symbol_ascii(decorator);
      }

      std::string
      get_symbol_latex(const SymbolicDecorations &decorator) const
      {
        return space.get_symbol_latex(decorator);
      }

      const ExtractorType &
      get_extractor() const
      {
        return extractor;
      }

    protected:
      // Allow access to get_space()
      friend WeakForms::SelfLinearization::internal::ConvertTo;

      // Only want this to be a base class providing common implementation
      // for concrete views
      explicit SubSpaceViewBase(const SpaceType &    space,
                                const ExtractorType &extractor)
        : space(space)
        , extractor(extractor)
      {}

      SubSpaceViewBase(const SubSpaceViewBase &) = default;

      const SpaceType &
      get_space() const
      {
        return space;
      }

    private:
      const SpaceType     space;
      const ExtractorType extractor;
    };


    template <typename SpaceType_>
    class Scalar final
      : public SubSpaceViewBase<SpaceType_, FEValuesExtractors::Scalar>
    {
      using Base_t = SubSpaceViewBase<SpaceType_, FEValuesExtractors::Scalar>;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = SpaceType_::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = SpaceType_::space_dimension;

      /**
       * Rank of subspace
       */
      static const int rank = 0;

      static_assert(rank == SpaceType_::rank,
                    "Unexpected rank in parent space.");

      using SpaceType = SpaceType_;

      using FEValuesExtractorType = typename Base_t::FEValuesExtractorType;

      using FEValuesViewsType =
        FEValuesViews::Scalar<dimension, space_dimension>;

      template <typename ScalarType>
      using OutputType =
        typename FEValuesViewsType::template OutputType<ScalarType>;

      template <typename ScalarType>
      using value_type = typename OutputType<ScalarType>::value_type;

      template <typename ScalarType>
      using gradient_type = typename OutputType<ScalarType>::gradient_type;

      template <typename ScalarType>
      using hessian_type = typename OutputType<ScalarType>::hessian_type;

      template <typename ScalarType>
      using laplacian_type = typename OutputType<ScalarType>::laplacian_type;

      template <typename ScalarType>
      using third_derivative_type =
        typename OutputType<ScalarType>::third_derivative_type;

      explicit Scalar(const SpaceType &                 space,
                      const FEValuesExtractors::Scalar &extractor)
        : Base_t(space, extractor)
      {}

      Scalar(const Scalar &) = default;

      virtual Scalar *
      clone() const override
      {
        return new Scalar(*this);
      }

      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::value(*this);
      }

      auto
      gradient() const
      {
        return WeakForms::gradient(*this);
      }

      auto
      laplacian() const
      {
        return WeakForms::laplacian(*this);
      }

      auto
      hessian() const
      {
        return WeakForms::hessian(*this);
      }

      auto
      third_derivative() const
      {
        return WeakForms::third_derivative(*this);
      }

      // Operators: Field solutions only

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::value<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      gradient() const
      {
        return WeakForms::gradient<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      laplacian() const
      {
        return WeakForms::laplacian<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      hessian() const
      {
        return WeakForms::hessian<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      third_derivative() const
      {
        return WeakForms::third_derivative<solution_index>(*this);
      }
    };


    template <typename SpaceType_>
    class Vector final
      : public SubSpaceViewBase<SpaceType_, FEValuesExtractors::Vector>
    {
      using Base_t = SubSpaceViewBase<SpaceType_, FEValuesExtractors::Vector>;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = SpaceType_::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = SpaceType_::space_dimension;

      /**
       * Rank of subspace
       */
      static const int rank = 1;

      using SpaceType = SpaceType_;

      using FEValuesExtractorType = typename Base_t::FEValuesExtractorType;

      using FEValuesViewsType =
        FEValuesViews::Vector<dimension, space_dimension>;

      template <typename ScalarType>
      using OutputType =
        typename FEValuesViewsType::template OutputType<ScalarType>;

      template <typename ScalarType>
      using value_type = typename OutputType<ScalarType>::value_type;

      template <typename ScalarType>
      using gradient_type = typename OutputType<ScalarType>::gradient_type;

      template <typename ScalarType>
      using symmetric_gradient_type =
        typename OutputType<ScalarType>::symmetric_gradient_type;

      template <typename ScalarType>
      using divergence_type = typename OutputType<ScalarType>::divergence_type;

      template <typename ScalarType>
      using curl_type = typename OutputType<ScalarType>::curl_type;

      template <typename ScalarType>
      using hessian_type = typename OutputType<ScalarType>::hessian_type;

      template <typename ScalarType>
      using third_derivative_type =
        typename OutputType<ScalarType>::third_derivative_type;

      explicit Vector(const SpaceType &                 space,
                      const FEValuesExtractors::Vector &extractor)
        : Base_t(space, extractor)
      {}

      Vector(const Vector &) = default;

      virtual Vector *
      clone() const override
      {
        return new Vector(*this);
      }

      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::value(*this);
      }

      auto
      gradient() const
      {
        return WeakForms::gradient(*this);
      }

      auto
      symmetric_gradient() const
      {
        return WeakForms::symmetric_gradient(*this);
      }

      auto
      divergence() const
      {
        return WeakForms::divergence(*this);
      }

      auto
      curl() const
      {
        return WeakForms::curl(*this);
      }

      auto
      hessian() const
      {
        return WeakForms::hessian(*this);
      }

      auto
      third_derivative() const
      {
        return WeakForms::third_derivative(*this);
      }

      // Operators: Field solutions only

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::value<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      gradient() const
      {
        return WeakForms::gradient<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      symmetric_gradient() const
      {
        return WeakForms::symmetric_gradient<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      divergence() const
      {
        return WeakForms::divergence<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      curl() const
      {
        return WeakForms::curl<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      hessian() const
      {
        return WeakForms::hessian<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      third_derivative() const
      {
        return WeakForms::third_derivative<solution_index>(*this);
      }
    };


    template <int rank_, typename SpaceType_>
    class Tensor final
      : public SubSpaceViewBase<SpaceType_, FEValuesExtractors::Tensor<rank_>>
    {
      using Base_t =
        SubSpaceViewBase<SpaceType_, FEValuesExtractors::Tensor<rank_>>;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = SpaceType_::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = SpaceType_::space_dimension;

      /**
       * Rank of subspace
       */
      static const int rank = rank_;

      using SpaceType = SpaceType_;

      using FEValuesExtractorType = typename Base_t::FEValuesExtractorType;

      using FEValuesViewsType = FEValuesViews::Tensor<rank_, space_dimension>;

      template <typename ScalarType>
      using OutputType =
        typename FEValuesViewsType::template OutputType<ScalarType>;

      template <typename ScalarType>
      using value_type = typename OutputType<ScalarType>::value_type;

      template <typename ScalarType>
      using gradient_type = typename OutputType<ScalarType>::gradient_type;

      template <typename ScalarType>
      using divergence_type = typename OutputType<ScalarType>::divergence_type;

      explicit Tensor(const SpaceType &                        space,
                      const FEValuesExtractors::Tensor<rank_> &extractor)
        : Base_t(space, extractor)
      {}

      Tensor(const Tensor &) = default;

      virtual Tensor *
      clone() const override
      {
        return new Tensor(*this);
      }

      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::value(*this);
      }

      auto
      gradient() const
      {
        return WeakForms::gradient(*this);
      }

      auto
      divergence() const
      {
        return WeakForms::divergence(*this);
      }

      // Operators: Field solutions only

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::value<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      gradient() const
      {
        return WeakForms::gradient<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      divergence() const
      {
        return WeakForms::divergence<solution_index>(*this);
      }
    };


    template <int rank_, typename SpaceType_>
    class SymmetricTensor final
      : public SubSpaceViewBase<SpaceType_,
                                FEValuesExtractors::SymmetricTensor<rank_>>
    {
      using Base_t =
        SubSpaceViewBase<SpaceType_,
                         FEValuesExtractors::SymmetricTensor<rank_>>;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = SpaceType_::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = SpaceType_::space_dimension;

      /**
       * Rank of subspace
       */
      static const int rank = rank_;

      using SpaceType = SpaceType_;

      using FEValuesExtractorType = typename Base_t::FEValuesExtractorType;

      using FEValuesViewsType =
        FEValuesViews::SymmetricTensor<rank_, space_dimension>;

      template <typename ScalarType>
      using OutputType =
        typename FEValuesViewsType::template OutputType<ScalarType>;

      template <typename ScalarType>
      using value_type = typename OutputType<ScalarType>::value_type;

      template <typename ScalarType>
      using divergence_type = typename OutputType<ScalarType>::divergence_type;

      explicit SymmetricTensor(
        const SpaceType &                                 space,
        const FEValuesExtractors::SymmetricTensor<rank_> &extractor)
        : Base_t(space, extractor)
      {}

      SymmetricTensor(const SymmetricTensor &) = default;

      virtual SymmetricTensor *
      clone() const override
      {
        return new SymmetricTensor(*this);
      }

      // Operators: Test functions, trial solutions, and field solutions

      auto
      value() const
      {
        return WeakForms::value(*this);
      }

      auto
      divergence() const
      {
        return WeakForms::divergence(*this);
      }

      // Operators: Field solutions only

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      value() const
      {
        return WeakForms::value<solution_index>(*this);
      }

      template <
        std::size_t solution_index = 0,
        typename T                 = SpaceType_,
        typename = typename std::enable_if<is_field_solution<T>::value>::type>
      auto
      divergence() const
      {
        return WeakForms::divergence<solution_index>(*this);
      }
    };

  } // namespace SubSpaceViews

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ---- Finite element spaces: Test functions and trial solutions ---- */

    /**
     * Extract the shape function values from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::value,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpValueBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpValueBase<SubSpaceViewsType>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].value(dof_index,
                                                                    q_point);
      }

      /**
       * Return all shape function values at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function gradients from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::gradient,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpGradientBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpGradientBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
          std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Tensor<View_t::rank, Space_t>>::value,
        "The selected subspace view does not support the gradient operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].gradient(
          dof_index, q_point);
      }

      /**
       * Return all shape function gradients at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function symmetric gradients from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::symmetric_gradient,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpSymmetricGradientBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpSymmetricGradientBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the symmetric gradient operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()]
          .symmetric_gradient(dof_index, q_point);
      }

      /**
       * Return all shape function symmetric gradients at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function divergences from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::divergence,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpDivergenceBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpDivergenceBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Tensor<View_t::rank, Space_t>>::value ||
          std::is_same<
            View_t,
            SubSpaceViews::SymmetricTensor<View_t::rank, Space_t>>::value,
        "The selected subspace view does not support the divergence operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].divergence(
          dof_index, q_point);
      }

      /**
       * Return all shape function divergences at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function divergences from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::curl,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpCurlBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpCurlBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the curls operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      // In dim==2, the curl operation returns a interestingly dimensioned
      // tensor that is not easily compatible with this framework.
      static_assert(
        dimension == 3,
        "The curl operation for the selected subspace view is only implemented in 3d.");

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].curl(dof_index,
                                                                   q_point);
      }

      /**
       * Return all shape function curls at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function Laplacians from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::laplacian,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpLaplacianBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpLaplacianBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value,
        "The selected subspace view does not support the Laplacian operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return trace(
          fe_values[this->get_operand().get_extractor()].hessian(dof_index,
                                                                 q_point));
      }

      /**
       * Return all shape function Laplacians at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function Hessians from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::hessian,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpHessianBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpHessianBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
          std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the Hessian operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].hessian(dof_index,
                                                                      q_point);
      }

      /**
       * Return all shape function Hessians at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /**
     * Extract the shape function third derivatives from a finite element
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class SymbolicOp<
      SubSpaceViewsType,
      SymbolicOpCodes::third_derivative,
      typename std::enable_if<
        is_test_function<typename SubSpaceViewsType::SpaceType>::value ||
        is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public SymbolicOpThirdDerivativeBase<SubSpaceViewsType>
    {
      using View_t  = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t  = SymbolicOpThirdDerivativeBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
          std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
        "The selected subspace view does not support the third derivative operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename ScalarType>
      value_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                              dof_index,
                 const unsigned int                              q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].third_derivative(
          dof_index, q_point);
      }

      /**
       * Return all shape function third derivatives at a quadrature point
       *
       * @tparam ScalarType
       * @param fe_values
       * @param q_point
       * @return return_type<ScalarType>
       */
      template <typename ScalarType>
      return_type<ScalarType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values_dofs,
                 const FEValuesBase<dimension, space_dimension> &fe_values_op,
                 const unsigned int                              q_point) const
      {
        Assert(q_point < fe_values_op.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values_op.n_quadrature_points));

        return_type<ScalarType> out;
        out.reserve(fe_values_dofs.dofs_per_cell);

        for (const auto &dof_index : fe_values_dofs.dof_indices())
          out.emplace_back(this->template operator()<ScalarType>(fe_values_op,
                                                                 dof_index,
                                                                 q_point));

        return out;
      }
    };



    /* ------------ Finite element spaces: Solution fields ------------ */

    namespace internal
    {
      template <typename SubSpaceViewsType, enum SymbolicOpCodes>
      struct SymbolicOpExtractor;


      // Scalar
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::Scalar;

        template <int spacedim>
        static constexpr unsigned int n_components = 1;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::gradient>
      {
        using type = FEValuesExtractors::Vector;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<1, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::hessian>
      {
        using type = FEValuesExtractors::Tensor<2>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<2, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::laplacian>
      {
        using type = FEValuesExtractors::Scalar;

        template <int spacedim>
        static constexpr unsigned int n_components = 1;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Scalar<SpaceType>,
                                 SymbolicOpCodes::third_derivative>
      {
        using type = FEValuesExtractors::Tensor<3>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<3, spacedim>::n_independent_components;
      };


      // Vector
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::Vector;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<1, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::gradient>
      {
        using type = FEValuesExtractors::Tensor<2>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<2, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::symmetric_gradient>
      {
        using type = FEValuesExtractors::SymmetricTensor<2>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          SymmetricTensor<2, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::divergence>
      {
        using type = FEValuesExtractors::Scalar;

        template <int spacedim>
        static constexpr unsigned int n_components = 1;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::curl>
      {
        using type = FEValuesExtractors::Vector;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<1, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::hessian>
      {
        using type = FEValuesExtractors::Tensor<3>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<3, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Vector<SpaceType>,
                                 SymbolicOpCodes::third_derivative>
      {
        using type = FEValuesExtractors::Tensor<4>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<4, spacedim>::n_independent_components;
      };


      // Tensor
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Tensor<2, SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::Tensor<2>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<2, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Tensor<2, SpaceType>,
                                 SymbolicOpCodes::divergence>
      {
        using type = FEValuesExtractors::Vector;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<1, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::Tensor<2, SpaceType>,
                                 SymbolicOpCodes::gradient>
      {
        using type = FEValuesExtractors::Tensor<3>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<3, spacedim>::n_independent_components;
      };


      // Symmetric Tensor
      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::SymmetricTensor<2, SpaceType>,
                                 SymbolicOpCodes::value>
      {
        using type = FEValuesExtractors::SymmetricTensor<2>;

        template <int spacedim>
        static constexpr unsigned int n_components =
          SymmetricTensor<2, spacedim>::n_independent_components;
      };

      template <typename SpaceType>
      struct SymbolicOpExtractor<SubSpaceViews::SymmetricTensor<2, SpaceType>,
                                 SymbolicOpCodes::divergence>
      {
        using type = FEValuesExtractors::Vector;

        template <int spacedim>
        static constexpr unsigned int n_components =
          Tensor<1, spacedim>::n_independent_components;
      };
    } // namespace internal


    /**
     * Extract the solution values from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::value,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpValueBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpValueBase<View_t, solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<
          SubSpaceViewsType,
          SymbolicOpCodes::value>::template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type =
        typename internal::SymbolicOpExtractor<SubSpaceViewsType,
                                               SymbolicOpCodes::value>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution values at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data.template get_values<ExtractorType, ScalarType>(
          solution_names[solution_index], this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_values_from_local_dof_values(solution_local_dof_values,
        //                                              out);
        // return out;
      }
    };



    /**
     * Extract the solution gradients from the disretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::gradient,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpGradientBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpGradientBase<View_t, solution_index>;
      using typename Base_t::Op;

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<
          SubSpaceViewsType,
          SymbolicOpCodes::gradient>::template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type =
        typename internal::SymbolicOpExtractor<SubSpaceViewsType,
                                               SymbolicOpCodes::gradient>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // using Base_t::as_ascii;
      // using Base_t::as_latex;

      // Return solution gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data.template get_gradients<ExtractorType, ScalarType>(
          solution_names[solution_index], this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_gradients_from_local_dof_values(
        //     solution_local_dof_values, out);
        // return out;
      }
    };



    /**
     * Extract the solution symmetric gradients from the disretised solution
     * field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::symmetric_gradient,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpSymmetricGradientBase<SubSpaceViewsType,
                                               solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpSymmetricGradientBase<View_t, solution_index>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<
          View_t,
          SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the symmetric gradient operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<SubSpaceViewsType,
                                      SymbolicOpCodes::symmetric_gradient>::
          template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type = typename internal::SymbolicOpExtractor<
        SubSpaceViewsType,
        SymbolicOpCodes::symmetric_gradient>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution symmetric gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data
          .template get_symmetric_gradients<ExtractorType, ScalarType>(
            solution_names[solution_index],
            this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_symmetric_gradients_from_local_dof_values(
        //     solution_local_dof_values, out);
        // return out;
      }
    };



    /**
     * Extract the solution divergences from the disretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::divergence,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpDivergenceBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpDivergenceBase<View_t, solution_index>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t,
                     SubSpaceViews::Vector<
                       typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Tensor<
                         View_t::rank,
                         typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::SymmetricTensor<
                         View_t::rank,
                         typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the divergence operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<
          SubSpaceViewsType,
          SymbolicOpCodes::divergence>::template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type = typename internal::SymbolicOpExtractor<
        SubSpaceViewsType,
        SymbolicOpCodes::divergence>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution divergences at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data.template get_divergences<ExtractorType, ScalarType>(
          solution_names[solution_index], this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_divergences_from_local_dof_values(
        //     solution_local_dof_values, out);
        // return out;
      }
    };



    /**
     * Extract the solution curls from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::curl,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpCurlBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpCurlBase<View_t, solution_index>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<
          View_t,
          SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the curl operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<
          SubSpaceViewsType,
          SymbolicOpCodes::curl>::template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type =
        typename internal::SymbolicOpExtractor<SubSpaceViewsType,
                                               SymbolicOpCodes::curl>::type;

      // In dim==2, the curl operation returns a interestingly dimensioned
      // tensor that is not easily compatible with this framework.
      static_assert(
        dimension == 3,
        "The curl operation for the selected subspace view is only implemented in 3d.");

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution symmetric gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data.template get_curls<ExtractorType, ScalarType>(
          solution_names[solution_index], this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_curls_from_local_dof_values(solution_local_dof_values,
        //                                             out);
        // return out;
      }
    };



    /**
     * Extract the solution Laplacians from the disretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::laplacian,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpLaplacianBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpLaplacianBase<View_t, solution_index>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<
          View_t,
          SubSpaceViews::Scalar<typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the Laplacian operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<
          SubSpaceViewsType,
          SymbolicOpCodes::laplacian>::template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type = typename internal::SymbolicOpExtractor<
        SubSpaceViewsType,
        SymbolicOpCodes::laplacian>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution Laplacian at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data.template get_laplacians<ExtractorType, ScalarType>(
          solution_names[solution_index], this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_laplacians_from_local_dof_values(
        //     solution_local_dof_values, out);
        // return out;
      }
    };



    /**
     * Extract the solution Hessians from the disretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::hessian,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpHessianBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpHessianBase<View_t, solution_index>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t,
                     SubSpaceViews::Scalar<
                       typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Vector<
                         typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the Hessian operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<
          SubSpaceViewsType,
          SymbolicOpCodes::hessian>::template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type =
        typename internal::SymbolicOpExtractor<SubSpaceViewsType,
                                               SymbolicOpCodes::hessian>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution symmetric gradients at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data.template get_hessians<ExtractorType, ScalarType>(
          solution_names[solution_index], this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_hessians_from_local_dof_values(
        //     solution_local_dof_values, out);
        // return out;
      }
    };



    /**
     * Extract the solution third derivatives from the disretised solution field
     * subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType, std::size_t solution_index>
    class SymbolicOp<SubSpaceViewsType,
                     SymbolicOpCodes::third_derivative,
                     typename std::enable_if<is_field_solution<
                       typename SubSpaceViewsType::SpaceType>::value>::type,
                     WeakForms::internal::SolutionIndex<solution_index>>
      : public SymbolicOpThirdDerivativeBase<SubSpaceViewsType, solution_index>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = SymbolicOpThirdDerivativeBase<View_t, solution_index>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(
        std::is_same<View_t,
                     SubSpaceViews::Scalar<
                       typename SubSpaceViewsType::SpaceType>>::value ||
          std::is_same<View_t,
                       SubSpaceViews::Vector<
                         typename SubSpaceViewsType::SpaceType>>::value,
        "The selected subspace view does not support the third derivative operation.");

    public:
      /**
       * Dimension in which this object operates.
       */
      static const unsigned int dimension = View_t::dimension;

      /**
       * Dimension of the subspace in which this object operates.
       */
      static const unsigned int space_dimension = View_t::space_dimension;

      /**
       * Number of independent components associated with this field.
       */
      static constexpr unsigned int n_components =
        internal::SymbolicOpExtractor<SubSpaceViewsType,
                                      SymbolicOpCodes::third_derivative>::
          template n_components<space_dimension>;

      /**
       * The extractor corresponding to the operation performed on the subspace
       */
      using extractor_type = typename internal::SymbolicOpExtractor<
        SubSpaceViewsType,
        SymbolicOpCodes::third_derivative>::type;

      template <typename ScalarType>
      using value_type = typename Base_t::template value_type<ScalarType>;
      template <typename ScalarType>
      using return_type = typename Base_t::template return_type<ScalarType>;

      explicit SymbolicOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution third derivatives at all quadrature points
      template <typename ScalarType>
      const return_type<ScalarType> &
      operator()(
        MeshWorker::ScratchData<dimension, space_dimension> &scratch_data,
        const std::vector<std::string> &solution_names) const
      {
        Assert(solution_index < solution_names.size(),
               ExcIndexRange(solution_index, 0, solution_names.size()));

        using ExtractorType = typename SubSpaceViewsType::FEValuesExtractorType;
        return scratch_data
          .template get_third_derivatives<ExtractorType, ScalarType>(
            solution_names[solution_index],
            this->get_operand().get_extractor());

        // return_type<ScalarType> out(fe_values.n_quadrature_points);
        // fe_values[this->get_operand().get_extractor()]
        //   .get_function_third_derivatives_from_local_dof_values(
        //     solution_local_dof_values, out);
        // return out;
      }
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /* ----- Finite element subspaces: Test functions and trial solutions ----- */

  /**
   * @brief Value variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

    return OpType(operand);
  }


  /**
   * @brief Value variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <int, class> typename SubSpaceViewsType,
            int rank,
            typename SpaceType>
  WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank, SpaceType>,
                                   WeakForms::Operators::SymbolicOpCodes::value>
  value(const SubSpaceViewsType<rank, SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<rank, SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

    return OpType(operand);
  }


  /**
   * @brief Gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::gradient>;

    return OpType(operand);
  }


  /**
   * @brief Gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <int, class> typename SubSpaceViewsType,
            int rank,
            typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::gradient>
  gradient(const SubSpaceViewsType<rank, SpaceType> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<rank, SpaceType>,
                   SubSpaceViews::Tensor<rank, SpaceType>>::value,
      "The selected subspace view does not support the gradient operation.");

    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<rank, SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::gradient>;

    return OpType(operand);
  }


  /**
   * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
  symmetric_gradient(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<SpaceType>,
                   SubSpaceViews::Vector<SpaceType>>::value,
      "The selected subspace view does not support the symmetric gradient operation.");

    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::symmetric_gradient>;

    return OpType(operand);
  }


  /**
   * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // typename SpaceType>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank, SpaceType>,
  //                               WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
  // symmetric_gradient(const SubSpaceViewsType<rank, SpaceType> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
  //   support the symmetric gradient operation.");

  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, SpaceType>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::symmetric_gradient>;

  //   return OpType(operand);
  // }


  /**
   * @brief Divergence variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::divergence>
  divergence(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<SpaceType>,
                   SubSpaceViews::Vector<SpaceType>>::value,
      "The selected subspace view does not support the divergence operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::divergence>;

    return OpType(operand);
  }


  /**
   * @brief Divergence variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <int, class> typename SubSpaceViewsType,
            int rank,
            typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::divergence>
  divergence(const SubSpaceViewsType<rank, SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<rank, SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::divergence>;

    return OpType(operand);
  }


  /**
   * @brief Curl variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
                                   WeakForms::Operators::SymbolicOpCodes::curl>
  curl(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<SpaceType>,
                   SubSpaceViews::Vector<SpaceType>>::value,
      "The selected subspace view does not support the curl operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::curl>;

    return OpType(operand);
  }


  /**
   * @brief Curl variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // typename SpaceType>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank, SpaceType>,
  //                               WeakForms::Operators::SymbolicOpCodes::curl>
  // curl(const SubSpaceViewsType<rank, SpaceType> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
  //   support the symmetric gradient operation.");

  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, SpaceType>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::curl>;

  //   return OpType(operand);
  // }


  /**
   * @brief Laplacian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::laplacian>
  laplacian(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<SpaceType>,
                   SubSpaceViews::Scalar<SpaceType>>::value,
      "The selected subspace view does not support the Laplacian operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

    return OpType(operand);
  }


  // /**
  //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  *
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand
  //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::SymbolicOpCodes::value>
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // typename SpaceType>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,SpaceType>,
  //                               WeakForms::Operators::SymbolicOpCodes::laplacian>
  // laplacian(const SubSpaceViewsType<rank,SpaceType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,SpaceType>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

  //   return OpType(operand);
  // }


  /**
   * @brief Hessian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::hessian>
  hessian(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

    return OpType(operand);
  }


  // /**
  //  * @brief Hessian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  *
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand
  //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::SymbolicOpCodes::value>
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // typename SpaceType>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,SpaceType>,
  //                               WeakForms::Operators::SymbolicOpCodes::hessian>
  // hessian(const SubSpaceViewsType<rank,SpaceType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,SpaceType>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

  //   return OpType(operand);
  // }


  /**
   * @brief Third derivative variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <template <class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<SpaceType>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative>
  third_derivative(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

    return OpType(operand);
  }


  // /**
  //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  *
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand
  //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::SymbolicOpCodes::value>
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // typename SpaceType>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,SpaceType>,
  //                               WeakForms::Operators::SymbolicOpCodes::third_derivative>
  // third_derivative(const SubSpaceViewsType<rank,SpaceType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,SpaceType>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

  //   return OpType(operand);
  // }



  /* ------------- Finite element subspaces: Field solutions ------------- */

  /**
   * @brief Value variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::value,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Value variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <int, class> typename SubSpaceViewsType,
            int rank,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::value,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  value(const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::value,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::gradient,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <int, class> typename SubSpaceViewsType,
            int rank,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  gradient(const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand)
  {
    static_assert(
      std::is_same<
        SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
        SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>>::value,
      "The selected subspace view does not support the gradient operation.");

    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::gradient,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::symmetric_gradient,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  symmetric_gradient(
    const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<FieldSolution<dim, spacedim>>,
                   SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>::value,
      "The selected subspace view does not support the symmetric gradient operation.");

    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::symmetric_gradient,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Symmetric gradient variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // int dim, int spacedim>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,
  // FieldSolution<dim,spacedim>>,
  //                               WeakForms::Operators::SymbolicOpCodes::symmetric_gradient>
  // symmetric_gradient(const SubSpaceViewsType<rank,
  // FieldSolution<dim,spacedim>> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
  //   support the symmetric gradient operation.");

  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, FieldSolution<dim,spacedim>>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::symmetric_gradient>;

  //   return OpType(operand);
  // }


  /**
   * @brief Divergence variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::divergence,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  divergence(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<FieldSolution<dim, spacedim>>,
                   SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>::value,
      "The selected subspace view does not support the divergence operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::divergence,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Divergence variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <int, class> typename SubSpaceViewsType,
            int rank,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::divergence,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  divergence(
    const SubSpaceViewsType<rank, FieldSolution<dim, spacedim>> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<rank, FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::divergence,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Curl variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::curl,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  curl(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<FieldSolution<dim, spacedim>>,
                   SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>::value,
      "The selected subspace view does not support the curl operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::curl,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  /**
   * @brief Curl variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // int dim, int spacedim>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,
  // FieldSolution<dim,spacedim>>,
  //                               WeakForms::Operators::SymbolicOpCodes::curl>
  // curl(const SubSpaceViewsType<rank, FieldSolution<dim,spacedim>> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not
  //   support the symmetric gradient operation.");

  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, FieldSolution<dim,spacedim>>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::curl>;

  //   return OpType(operand);
  // }


  /**
   * @brief Laplacian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::laplacian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  laplacian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    static_assert(
      std::is_same<SubSpaceViewsType<FieldSolution<dim, spacedim>>,
                   SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>>::value,
      "The selected subspace view does not support the Laplacian operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::laplacian,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  // /**
  //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  *
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand
  //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::SymbolicOpCodes::value>
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // int dim, int spacedim>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>,
  //                               WeakForms::Operators::SymbolicOpCodes::laplacian>
  // laplacian(const SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>
  // &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::laplacian>;

  //   return OpType(operand);
  // }


  /**
   * @brief Hessian variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::hessian,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  hessian(const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::hessian,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  // /**
  //  * @brief Hessian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  *
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand
  //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::SymbolicOpCodes::value>
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // int dim, int spacedim>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>,
  //                               WeakForms::Operators::SymbolicOpCodes::hessian>
  // hessian(const SubSpaceViewsType<rank,FieldSolution<dim,spacedim>> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::hessian>;

  //   return OpType(operand);
  // }


  /**
   * @brief Third derivative variant for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   *
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand
   * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::SymbolicOpCodes::value>
   */
  template <std::size_t solution_index,
            template <class> typename SubSpaceViewsType,
            int dim,
            int spacedim>
  WeakForms::Operators::SymbolicOp<
    SubSpaceViewsType<FieldSolution<dim, spacedim>>,
    WeakForms::Operators::SymbolicOpCodes::third_derivative,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>
  third_derivative(
    const SubSpaceViewsType<FieldSolution<dim, spacedim>> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = SubSpaceViewsType<FieldSolution<dim, spacedim>>;
    using OpType =
      SymbolicOp<Op,
                 SymbolicOpCodes::third_derivative,
                 void,
                 WeakForms::internal::SolutionIndex<solution_index>>;

    return OpType(operand);
  }


  // /**
  //  * @brief Laplacian variant for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  *
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand
  //  * @return WeakForms::Operators::SymbolicOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::SymbolicOpCodes::value>
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank,
  // int dim, int spacedim>
  // WeakForms::Operators::SymbolicOp<SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>,
  //                               WeakForms::Operators::SymbolicOpCodes::third_derivative>
  // third_derivative(const SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>
  // &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,FieldSolution<dim,spacedim>>;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::third_derivative>;

  //   return OpType(operand);
  // }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Subspace views

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<
    SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>>
    : std::true_type
  {};



  // Decorators

  template <int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Vector<TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_test_function<
    SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_trial_solution<
    SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_field_solution<
    SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_test_function<
    SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_trial_solution<
    SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>>
    : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_field_solution<
    SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>>
    : std::true_type
  {};



  // Unary operations: Subspace views

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<
    Operators::SymbolicOp<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            std::size_t                     solution_index>
  struct is_field_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>,
                          OpCode,
                          void,
                          WeakForms::internal::SolutionIndex<solution_index>>>
    : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<
    Operators::SymbolicOp<SubSpaceViews::Vector<TestFunction<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>,
                          OpCode>> : std::true_type
  {};

  template <int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            std::size_t                     solution_index>
  struct is_field_solution_op<
    Operators::SymbolicOp<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>,
                          OpCode,
                          void,
                          WeakForms::internal::SolutionIndex<solution_index>>>
    : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<Operators::SymbolicOp<
    SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<Operators::SymbolicOp<
    SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            std::size_t                     solution_index>
  struct is_field_solution_op<Operators::SymbolicOp<
    SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>,
    OpCode,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_test_function_op<Operators::SymbolicOp<
    SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_trial_solution_op<Operators::SymbolicOp<
    SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>,
    OpCode>> : std::true_type
  {};

  template <int                             rank,
            int                             dim,
            int                             spacedim,
            enum Operators::SymbolicOpCodes OpCode,
            std::size_t                     solution_index>
  struct is_field_solution_op<Operators::SymbolicOp<
    SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>,
    OpCode,
    void,
    WeakForms::internal::SolutionIndex<solution_index>>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
