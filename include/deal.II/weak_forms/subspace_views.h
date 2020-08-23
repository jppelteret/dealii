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

#include <deal.II/weak_forms/spaces.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>


DEAL_II_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  /* --------------- Finite element subspaces --------------- */
  
  
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const SubSpaceViewsType<SpaceType> &operand);


  template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const SubSpaceViewsType<rank,SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const SubSpaceViewsType<SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::symmetric_gradient>
  symmetric_gradient(const SubSpaceViewsType<SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::divergence>
  divergence(const SubSpaceViewsType<SpaceType> &operand);


  template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::divergence>
  divergence(const SubSpaceViewsType<rank,SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::curl>
  curl(const SubSpaceViewsType<SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::laplacian>
  laplacian(const SubSpaceViewsType<SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::hessian>
  hessian(const SubSpaceViewsType<SpaceType> &operand);


  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::third_derivative>
  third_derivative(const SubSpaceViewsType<SpaceType> &operand);

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
      // Only want this to be a base class providing common implementation
      // for concrete views
      explicit SubSpaceViewBase(const SpaceType &space,
                       const ExtractorType &extractor)
        : space(space)
        , extractor(extractor)
      {}

    private:
      const SpaceType space;
      const ExtractorType extractor;
    };


    template <typename SpaceType_>
    class Scalar final : public SubSpaceViewBase<SpaceType_,FEValuesExtractors::Scalar>
    {
      using Base_t = SubSpaceViewBase<SpaceType_,FEValuesExtractors::Scalar>;

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
      
      static_assert(rank == SpaceType_::rank, "Unexpected rank in parent space.");

      using SpaceType = SpaceType_;

      using FEValuesExtractorType = typename Base_t::FEValuesExtractorType;

      using FEValuesViewsType = FEValuesViews::Scalar<dimension, space_dimension>;

      template <typename NumberType>
      using OutputType = typename FEValuesViewsType::template OutputType<NumberType>;

      template <typename NumberType>
      using value_type = typename OutputType<NumberType>::value_type;

      template <typename NumberType>
      using gradient_type = typename OutputType<NumberType>::gradient_type;

      template <typename NumberType>
      using hessian_type = typename OutputType<NumberType>::hessian_type;

      template <typename NumberType>
      using laplacian_type = typename OutputType<NumberType>::laplacian_type;

      template <typename NumberType>
      using third_derivative_type = typename OutputType<NumberType>::third_derivative_type;

      explicit Scalar(const SpaceType &space,
             const FEValuesExtractors::Scalar &extractor)
        : Base_t(space, extractor)
      {}

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
    };


    template <typename SpaceType_>
    class Vector final : public SubSpaceViewBase<SpaceType_,FEValuesExtractors::Vector>
    {
      using Base_t = SubSpaceViewBase<SpaceType_,FEValuesExtractors::Vector>;

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

      using FEValuesViewsType = FEValuesViews::Vector<dimension, space_dimension>;

      template <typename NumberType>
      using OutputType = typename FEValuesViewsType::template OutputType<NumberType>;

      template <typename NumberType>
      using value_type = typename OutputType<NumberType>::value_type;

      template <typename NumberType>
      using gradient_type = typename OutputType<NumberType>::gradient_type;

      template <typename NumberType>
      using symmetric_gradient_type = typename OutputType<NumberType>::symmetric_gradient_type;

      template <typename NumberType>
      using divergence_type = typename OutputType<NumberType>::divergence_type;

      template <typename NumberType>
      using curl_type = typename OutputType<NumberType>::curl_type;

      template <typename NumberType>
      using hessian_type = typename OutputType<NumberType>::hessian_type;

      template <typename NumberType>
      using third_derivative_type = typename OutputType<NumberType>::third_derivative_type;

      explicit Vector(const SpaceType &space,
             const FEValuesExtractors::Vector &extractor)
        : Base_t(space, extractor)
      {}

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
    };


    template <int rank_, typename SpaceType_>
    class Tensor final : public SubSpaceViewBase<SpaceType_,FEValuesExtractors::Tensor<rank_>>
    {
      using Base_t = SubSpaceViewBase<SpaceType_,FEValuesExtractors::Tensor<rank_>>;

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

      template <typename NumberType>
      using OutputType = typename FEValuesViewsType::template OutputType<NumberType>;

      template <typename NumberType>
      using value_type = typename OutputType<NumberType>::value_type;

      template <typename NumberType>
      using gradient_type = typename OutputType<NumberType>::gradient_type;

      template <typename NumberType>
      using divergence_type = typename OutputType<NumberType>::divergence_type;

      explicit Tensor(const SpaceType &space,
             const FEValuesExtractors::Tensor<rank_> &extractor)
        : Base_t(space, extractor)
      {}

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
    };


    template <int rank_, typename SpaceType_>
    class SymmetricTensor final : public SubSpaceViewBase<SpaceType_,FEValuesExtractors::SymmetricTensor<rank_>>
    {
      using Base_t = SubSpaceViewBase<SpaceType_,FEValuesExtractors::SymmetricTensor<rank_>>;

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

      using FEValuesViewsType = FEValuesViews::SymmetricTensor<rank_, space_dimension>;

      template <typename NumberType>
      using OutputType = typename FEValuesViewsType::template OutputType<NumberType>;

      template <typename NumberType>
      using value_type = typename OutputType<NumberType>::value_type;

      template <typename NumberType>
      using divergence_type = typename OutputType<NumberType>::divergence_type;

      explicit SymmetricTensor(const SpaceType &space,
             const FEValuesExtractors::SymmetricTensor<rank_> &extractor)
        : Base_t(space, extractor)
      {}

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
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::value,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpValueBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpValueBase<SubSpaceViewsType>;
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].value(dof_index, q_point);
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
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

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
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::gradient,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpGradientBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpGradientBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
                    std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].gradient(dof_index, q_point);
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
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

        return out;
      }
    };



    /**
     * Extract the shape function symmetric gradients from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::symmetric_gradient,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpSymmetricGradientBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpSymmetricGradientBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].symmetric_gradient(dof_index, q_point);
      }

      /**
       * Return all shape function symmetric gradients at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

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
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::divergence,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpDivergenceBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpDivergenceBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value ||
                    std::is_same<View_t, SubSpaceViews::Tensor<View_t::rank,Space_t>>::value ||
                    std::is_same<View_t, SubSpaceViews::SymmetricTensor<View_t::rank,Space_t>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].divergence(dof_index, q_point);
      }

      /**
       * Return all shape function divergences at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

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
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::curl,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpCurlBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpCurlBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Vector<Space_t>>::value,
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

      // In dim==2, the curl operation returns a interestingly dimensioned tensor that is
      // not easily compatible with this framework. 
      static_assert(dimension == 3, "The curl operation for the selected subspace view is only implemented in 3d.");

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].curl(dof_index, q_point);
      }

      /**
       * Return all shape function curls at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

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
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::laplacian,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpLaplacianBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpLaplacianBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return trace(fe_values[this->get_operand().get_extractor()].hessian(dof_index, q_point));
      }

      /**
       * Return all shape function Laplacians at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

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
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::hessian,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpHessianBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpHessianBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].hessian(dof_index, q_point);
      }

      /**
       * Return all shape function Hessians at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

        return out;
      }
    };



    /**
     * Extract the shape function third derivatives from a finite element subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Vector
     * @tparam SpaceType A space type, specifically a test space or trial space
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::third_derivative,
             typename std::enable_if<is_test_function<typename SubSpaceViewsType::SpaceType>::value || 
                                     is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpThirdDerivativeBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Space_t = typename View_t::SpaceType;
      using Base_t = UnaryOpThirdDerivativeBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<Space_t>>::value ||
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values[this->get_operand().get_extractor()].third_derivative(dof_index, q_point);
      }

      /**
       * Return all shape function third derivatives at a quadrature point
       *
       * @tparam NumberType
       * @param fe_values
       * @param q_point
       * @return return_type<NumberType>
       */
      template <typename NumberType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.n_quadrature_points);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

        return out;
      }
    };



    /* ------------ Finite element spaces: Solution fields ------------ */


    /**
     * Extract the solution values from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::value,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpValueBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpValueBase<View_t>;
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution values at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_values(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution gradients from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::gradient,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpGradientBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpGradientBase<View_t>;
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_gradients(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution symmetric gradients from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::symmetric_gradient,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpSymmetricGradientBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpSymmetricGradientBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution symmetric gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_symmetric_gradients(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution divergences from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::divergence,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpDivergenceBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpDivergenceBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value ||
                    std::is_same<View_t, SubSpaceViews::Tensor<View_t::rank, typename SubSpaceViewsType::SpaceType>>::value ||
                    std::is_same<View_t, SubSpaceViews::SymmetricTensor<View_t::rank, typename SubSpaceViewsType::SpaceType>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution divergences at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_divergences(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution curls from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::curl,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpCurlBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpCurlBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
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

      // In dim==2, the curl operation returns a interestingly dimensioned tensor that is
      // not easily compatible with this framework. 
      static_assert(dimension == 3, "The curl operation for the selected subspace view is only implemented in 3d.");

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution symmetric gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_curls(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution Laplacians from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::laplacian,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpLaplacianBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpLaplacianBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<typename SubSpaceViewsType::SpaceType>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution Laplacian at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_laplacians(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution Hessians from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::hessian,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpHessianBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpHessianBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<typename SubSpaceViewsType::SpaceType>>::value ||
                    std::is_same<View_t, SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution symmetric gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_hessians(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution third derivatives from the disretised solution field subspace.
     *
     * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
     * @tparam SpaceType A space type, specifically a solution field
     */
    template <typename SubSpaceViewsType>
    class UnaryOp<SubSpaceViewsType, UnaryOpCodes::third_derivative,
             typename std::enable_if<is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
      : public UnaryOpThirdDerivativeBase<SubSpaceViewsType>
    {
      using View_t = SubSpaceViewsType;
      using Base_t = UnaryOpThirdDerivativeBase<View_t>;
      using typename Base_t::Op;

      // Let's make any compilation failures due to template mismatches
      // easier to understand.
      static_assert(std::is_same<View_t, SubSpaceViews::Scalar<typename SubSpaceViewsType::SpaceType>>::value ||
                    std::is_same<View_t, SubSpaceViews::Vector<typename SubSpaceViewsType::SpaceType>>::value,
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

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
      
      // Return solution third derivatives at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dimension, space_dimension> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values[this->get_operand().get_extractor()].get_function_third_derivatives(solution, out);
        return out;
      }
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /* --------------- Finite element subspaces --------------- */

  /**
   * @brief Value varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }


  /**
   * @brief Value varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const SubSpaceViewsType<rank,SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<rank,SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }


  /**
   * @brief Gradient varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }


  /**
   * @brief Gradient varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  // WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank, SpaceType>,
  //                               WeakForms::Operators::UnaryOpCodes::gradient>
  // gradient(const SubSpaceViewsType<rank, SpaceType> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not support the gradient operation.");
    
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, SpaceType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

  //   return OpType(operand);
  // }


  /**
   * @brief Symmetric gradient varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::symmetric_gradient>
  symmetric_gradient(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(std::is_same<SubSpaceViewsType<SpaceType>, SubSpaceViews::Vector<SpaceType>>::value,
                  "The selected subspace view does not support the symmetric gradient operation.");

    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::symmetric_gradient>;

    return OpType(operand);
  }


  /**
   * @brief Symmetric gradient varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  // WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank, SpaceType>,
  //                               WeakForms::Operators::UnaryOpCodes::symmetric_gradient>
  // symmetric_gradient(const SubSpaceViewsType<rank, SpaceType> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not support the symmetric gradient operation.");
    
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, SpaceType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::symmetric_gradient>;

  //   return OpType(operand);
  // }


  /**
   * @brief Divergence varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::divergence>
  divergence(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(std::is_same<SubSpaceViewsType<SpaceType>, SubSpaceViews::Vector<SpaceType>>::value,
                  "The selected subspace view does not support the divergence operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::divergence>;

    return OpType(operand);
  }


  /**
   * @brief Divergence varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::divergence>
  divergence(const SubSpaceViewsType<rank,SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<rank,SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::divergence>;

    return OpType(operand);
  }


  /**
   * @brief Curl varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::curl>
  curl(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(std::is_same<SubSpaceViewsType<SpaceType>, SubSpaceViews::Vector<SpaceType>>::value,
                  "The selected subspace view does not support the curl operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::curl>;

    return OpType(operand);
  }


  /**
   * @brief Curl varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType, e.g. WeakForms::SubSpaceViews::Scalar
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  // template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  // WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank, SpaceType>,
  //                               WeakForms::Operators::UnaryOpCodes::curl>
  // curl(const SubSpaceViewsType<rank, SpaceType> &operand)
  // {
  //   static_assert(false, "Tensor and SymmetricTensor subspace views do not support the symmetric gradient operation.");
    
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank, SpaceType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::curl>;

  //   return OpType(operand);
  // }


  /**
   * @brief Laplacian varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::laplacian>
  laplacian(const SubSpaceViewsType<SpaceType> &operand)
  {
    static_assert(std::is_same<SubSpaceViewsType<SpaceType>, SubSpaceViews::Scalar<SpaceType>>::value,
                  "The selected subspace view does not support the Laplacian operation.");
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::laplacian>;

    return OpType(operand);
  }


  // /**
  //  * @brief Laplacian varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  * 
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand 
  //  * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::UnaryOpCodes::value> 
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  // WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
  //                               WeakForms::Operators::UnaryOpCodes::laplacian>
  // laplacian(const SubSpaceViewsType<rank,SpaceType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,SpaceType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::laplacian>;

  //   return OpType(operand);
  // }


  /**
   * @brief Hessian varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::hessian>
  hessian(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::hessian>;

    return OpType(operand);
  }


  // /**
  //  * @brief Hessian varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  * 
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand 
  //  * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::UnaryOpCodes::value> 
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  // WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
  //                               WeakForms::Operators::UnaryOpCodes::hessian>
  // hessian(const SubSpaceViewsType<rank,SpaceType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,SpaceType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::hessian>;

  //   return OpType(operand);
  // }


  /**
   * @brief Third derivative varient for WeakForms::SubSpaceViews::Scalar, WeakForms::SubSpaceViews::Vector
   * 
   * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
   * @tparam SpaceType A space type, specifically a test space or trial space
   * @param operand 
   * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
   * WeakForms::Operators::UnaryOpCodes::value> 
   */
  template <template<class> typename SubSpaceViewsType, typename SpaceType>
  WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
                                WeakForms::Operators::UnaryOpCodes::third_derivative>
  third_derivative(const SubSpaceViewsType<SpaceType> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SubSpaceViewsType<SpaceType>;
    using OpType = UnaryOp<Op, UnaryOpCodes::third_derivative>;

    return OpType(operand);
  }


  // /**
  //  * @brief Laplacian varient for WeakForms::SubSpaceViews::Tensor, WeakForms::SubSpaceViews::SymmetricTensor
  //  * 
  //  * @tparam SubSpaceViewsType The type of view being applied to the SpaceType.
  //  * @tparam SpaceType A space type, specifically a test space or trial space
  //  * @param operand 
  //  * @return WeakForms::Operators::UnaryOp<SubSpaceViewsType<SpaceType>,
  //  * WeakForms::Operators::UnaryOpCodes::value> 
  //  */
  // template <template<int, class> typename SubSpaceViewsType, int rank, typename SpaceType>
  // WeakForms::Operators::UnaryOp<SubSpaceViewsType<rank,SpaceType>,
  //                               WeakForms::Operators::UnaryOpCodes::third_derivative>
  // third_derivative(const SubSpaceViewsType<rank,SpaceType> &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = SubSpaceViewsType<rank,SpaceType>;
  //   using OpType = UnaryOp<Op, UnaryOpCodes::third_derivative>;

  //   return OpType(operand);
  // }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Subspace views

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_subspace_view<SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};



  // Decorators

  template <int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Vector<TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_test_function<SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_test_function<SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_trial_solution<SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>> : std::true_type
  {};

  template <int rank, int dim, int spacedim>
  struct is_field_solution<SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>> : std::true_type
  {};



  // Unary operations: Subspace views

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
    Operators::UnaryOp<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
    Operators::UnaryOp<SubSpaceViews::Scalar<TrialSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
    Operators::UnaryOp<SubSpaceViews::Scalar<FieldSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
    Operators::UnaryOp<SubSpaceViews::Vector<TestFunction<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
    Operators::UnaryOp<SubSpaceViews::Vector<TrialSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
    Operators::UnaryOp<SubSpaceViews::Vector<FieldSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int rank, int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
    Operators::UnaryOp<SubSpaceViews::Tensor<rank, TestFunction<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int rank, int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
    Operators::UnaryOp<SubSpaceViews::Tensor<rank, TrialSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int rank, int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
    Operators::UnaryOp<SubSpaceViews::Tensor<rank, FieldSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int rank, int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
    Operators::UnaryOp<SubSpaceViews::SymmetricTensor<rank, TestFunction<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int rank, int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
    Operators::UnaryOp<SubSpaceViews::SymmetricTensor<rank, TrialSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

  template <int rank, int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
    Operators::UnaryOp<SubSpaceViews::SymmetricTensor<rank, FieldSolution<dim, spacedim>>, OpCode>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
