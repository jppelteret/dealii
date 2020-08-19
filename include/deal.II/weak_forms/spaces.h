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

#ifndef dealii_weakforms_spaces_h
#define dealii_weakforms_spaces_h

#include <deal.II/base/config.h>

// TODO: Move FeValuesViews::[Scalar/Vector/...]::Output<> into another header??
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/weak_forms/subspace_extractors.h>
#include <deal.II/weak_forms/symbolic_decorations.h>
#include <deal.II/weak_forms/type_traits.h>
#include <deal.II/weak_forms/unary_operators.h>


DEAL_II_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  template <int dim, int spacedim = dim>
  class TestFunction;
  template <int dim, int spacedim = dim>
  class TrialSolution;
  template <int dim, int spacedim = dim>
  class FieldSolution;

  namespace SubSpaceViews
  {
    template <typename SpaceType>
    class Scalar;
    template <typename SpaceType>
    class Vector;
    template <int rank, typename SpaceType>
    class Tensor;
    template <int rank, typename SpaceType>
    class SymmetricTensor;
  } // namespace SubSpaceViews

  /* --------------- Finite element spaces: Test functions --------------- */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TestFunction<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TestFunction<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TestFunction<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TestFunction<dim, spacedim> &operand);



  /* --------------- Finite element spaces: Trial solutions --------------- */



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TrialSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TrialSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TrialSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TrialSolution<dim, spacedim> &operand);



  /* --------------- Finite element spaces: Solution fields --------------- */



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::FieldSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand);



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::FieldSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand);

} // namespace WeakForms

#endif // DOXYGEN


namespace WeakForms
{
  template <int dim, int spacedim>
  class Space
  {
  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = dim;

    /**
     * Dimension of the space in which this object operates.
     */
    static const unsigned int space_dimension = spacedim;

    /**
     * Rank of continuous space
     */
    static const int rank = 0;

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

    // ----  Ascii ----

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.unary_op_operand_as_ascii(*this);
    }

    virtual std::string
    get_field_ascii(const SymbolicDecorations &decorator) const
    {
      return field_ascii;
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const = 0;

    // ---- LaTeX ----

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.unary_op_operand_as_latex(*this);
    }

    virtual std::string
    get_field_latex(const SymbolicDecorations &decorator) const
    {
      return field_latex;
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const = 0;

  protected:
    // Create a subspace
    Space(const std::string &        field_ascii,
          const std::string &        field_latex)
      : field_ascii(field_ascii)
      , field_latex(field_latex != "" ? field_latex : field_ascii)
    {}

    const std::string field_ascii;
    const std::string field_latex;
  };



  template <int dim, int spacedim>
  class TestFunction : public Space<dim, spacedim>
  {
  public:
    // Full space
    TestFunction()
      : TestFunction("","")
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

    std::string
    get_field_ascii(const SymbolicDecorations &decorator) const override
    {
      if (this->field_ascii.empty())
        return decorator.naming_ascii.solution_field;
      else
        return this->field_ascii;
    }

    std::string
    get_field_latex(const SymbolicDecorations &decorator) const override
    {
      if (this->field_latex.empty())
        return decorator.naming_latex.solution_field;
      else
        return this->field_latex;
    }

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.test_function;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.test_function;
    }

    SubSpaceViews::Scalar<TestFunction>
    operator[](const SubSpaceExtractors::Scalar &extractor) const
    {
      const TestFunction subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Scalar<TestFunction>(subspace, extractor.extractor);
    }

    SubSpaceViews::Vector<TestFunction>
    operator[](const SubSpaceExtractors::Vector &extractor) const
    {
      const TestFunction subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Vector<TestFunction>(subspace, extractor.extractor);
    }

    template<int rank>
    SubSpaceViews::Tensor<rank, TestFunction>
    operator[](const SubSpaceExtractors::Tensor<rank> &extractor) const
    {
      const TestFunction subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Tensor<rank, TestFunction>(subspace, extractor.extractor);
    }

    template<int rank>
    SubSpaceViews::SymmetricTensor<rank, TestFunction>
    operator[](const SubSpaceExtractors::SymmetricTensor<rank> &extractor) const
    {
      const TestFunction subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::SymmetricTensor<rank, TestFunction>(subspace, extractor.extractor);
    }

  protected:
    // Subspace
    TestFunction(const std::string &field_ascii,
                 const std::string &field_latex)
      : Space<dim, spacedim>(field_ascii,
                             field_latex)
    {}
  };



  template <int dim, int spacedim>
  class TrialSolution : public Space<dim, spacedim>
  {
  public:
    // Full space
    TrialSolution()
      : TrialSolution("","")
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

    std::string
    get_field_ascii(const SymbolicDecorations &decorator) const override
    {
      if (this->field_ascii.empty())
        return decorator.naming_ascii.solution_field;
      else
        return this->field_ascii;
    }

    std::string
    get_field_latex(const SymbolicDecorations &decorator) const override
    {
      if (this->field_latex.empty())
        return decorator.naming_latex.solution_field;
      else
        return this->field_latex;
    }

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.trial_solution;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.trial_solution;
    }

    SubSpaceViews::Scalar<TrialSolution>
    operator[](const SubSpaceExtractors::Scalar &extractor) const
    {
      const TrialSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Scalar<TrialSolution>(subspace, extractor.extractor);
    }

    SubSpaceViews::Vector<TrialSolution>
    operator[](const SubSpaceExtractors::Vector &extractor) const
    {
      const TrialSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Vector<TrialSolution>(subspace, extractor.extractor);
    }

    template<int rank>
    SubSpaceViews::Tensor<rank, TrialSolution>
    operator[](const SubSpaceExtractors::Tensor<rank> &extractor) const
    {
      const TrialSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Tensor<rank, TrialSolution>(subspace, extractor.extractor);
    }

    template<int rank>
    SubSpaceViews::SymmetricTensor<rank, TrialSolution>
    operator[](const SubSpaceExtractors::SymmetricTensor<rank> &extractor) const
    {
      const TrialSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::SymmetricTensor<rank, TrialSolution>(subspace, extractor.extractor);
    }

  protected:
    // Subspace
    TrialSolution(const std::string &field_ascii,
                  const std::string &field_latex)
      : Space<dim, spacedim>(field_ascii,
                             field_latex)
    {}
  };



  template <int dim, int spacedim>
  class FieldSolution : public Space<dim, spacedim>
  {
  public:
    // Full space
    FieldSolution()
      : FieldSolution("", "")
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

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_ascii.solution_field;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      return decorator.naming_latex.solution_field;
    }

    SubSpaceViews::Scalar<FieldSolution>
    operator[](const SubSpaceExtractors::Scalar &extractor) const
    {
      const FieldSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Scalar<FieldSolution>(subspace, extractor.extractor);
    }

    SubSpaceViews::Vector<FieldSolution>
    operator[](const SubSpaceExtractors::Vector &extractor) const
    {
      const FieldSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Vector<FieldSolution>(subspace, extractor.extractor);
    }

    template<int rank>
    SubSpaceViews::Tensor<rank, FieldSolution>
    operator[](const SubSpaceExtractors::Tensor<rank> &extractor) const
    {
      const FieldSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::Tensor<rank, FieldSolution>(subspace, extractor.extractor);
    }

    template<int rank>
    SubSpaceViews::SymmetricTensor<rank, FieldSolution>
    operator[](const SubSpaceExtractors::SymmetricTensor<rank> &extractor) const
    {
      const FieldSolution subspace(extractor.field_ascii, extractor.field_latex);
      return SubSpaceViews::SymmetricTensor<rank, FieldSolution>(subspace, extractor.extractor);
    }

  protected:
    // Subspace
    FieldSolution(const std::string          &field_ascii,
                  const std::string          &field_latex)
      : Space<dim, spacedim>(field_ascii,
                             field_latex)
    {}
  };



  // namespace Linear
  // {
  //   template <int dim, int spacedim = dim>
  //   using TestFunction = WeakForms::TestFunction<dim, spacedim>;

  //   template <int dim, int spacedim = dim>
  //   using TrialSolution = WeakForms::TrialSolution<dim, spacedim>;

  //   template <int dim, int spacedim = dim>
  //   using Solution = WeakForms::Solution<dim, spacedim>;
  // } // namespace Linear



  namespace NonLinear
  {
    template <int dim, int spacedim = dim>
    using Variation = WeakForms::TestFunction<dim, spacedim>;

    template <int dim, int spacedim = dim>
    using Linearization = WeakForms::TrialSolution<dim, spacedim>;

    template <int dim, int spacedim = dim>
    using FieldSolution = WeakForms::FieldSolution<dim, spacedim>;
  } // namespace NonLinear



  namespace SubSpaceViews
  {
    template <typename SpaceType_>
    class Scalar
    {
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

      Scalar(const SpaceType &space,
             const FEValuesExtractors::Scalar &extractor)
        : space(space)
        , extractor(extractor)
      {}

      // auto
      // value() const
      // {
      //   return WeakForms::value(*this);
      // }

      // auto
      // gradient() const
      // {
      //   return WeakForms::gradient(*this);
      // }
    
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

    private:
      const SpaceType space;
      const FEValuesExtractors::Scalar extractor;
    };


    template <typename SpaceType_>
    class Vector
    {
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

      Vector(const SpaceType &space,
             const FEValuesExtractors::Vector &extractor)
        : space(space)
        , extractor(extractor)
      {}

      // auto
      // value() const
      // {
      //   return WeakForms::value(*this);
      // }

      // auto
      // gradient() const
      // {
      //   return WeakForms::gradient(*this);
      // }
    
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

    private:
      const SpaceType space;
      const FEValuesExtractors::Vector extractor;
    };


    template <int rank_, typename SpaceType_>
    class Tensor
    {
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

      using FEValuesViewsType = FEValuesViews::Tensor<dimension, space_dimension>;

      template <typename NumberType>
      using OutputType = typename FEValuesViewsType::template OutputType<NumberType>;

      template <typename NumberType>
      using value_type = typename OutputType<NumberType>::value_type;

      template <typename NumberType>
      using divergence_type = typename OutputType<NumberType>::divergence_type;

      template <typename NumberType>
      using gradient_type = typename OutputType<NumberType>::gradient_type;

      Tensor(const SpaceType &space,
             const FEValuesExtractors::Tensor<rank_> &extractor)
        : space(space)
        , extractor(extractor)
      {}

      // auto
      // value() const
      // {
      //   return WeakForms::value(*this);
      // }

      // auto
      // gradient() const
      // {
      //   return WeakForms::gradient(*this);
      // }
    
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

    private:
      const SpaceType space;
      const FEValuesExtractors::Tensor<rank_> extractor;
    };


    template <int rank_, typename SpaceType_>
    class SymmetricTensor
    {
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

      using FEValuesViewsType = FEValuesViews::SymmetricTensor<dimension, space_dimension>;

      template <typename NumberType>
      using OutputType = typename FEValuesViewsType::template OutputType<NumberType>;

      template <typename NumberType>
      using value_type = typename OutputType<NumberType>::value_type;

      template <typename NumberType>
      using divergence_type = typename OutputType<NumberType>::divergence_type;

      SymmetricTensor(const SpaceType &space,
             const FEValuesExtractors::SymmetricTensor<rank_> &extractor)
        : space(space)
        , extractor(extractor)
      {}

      // auto
      // value() const
      // {
      //   return WeakForms::value(*this);
      // }

      // auto
      // gradient() const
      // {
      //   return WeakForms::gradient(*this);
      // }
    
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

    private:
      const SpaceType space;
      const FEValuesExtractors::SymmetricTensor<rank_> extractor;
    };
  } // namespace SubSpaceViews

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /* ---- Mix-in classes ---- */
    template<typename Op_>
    class UnaryOpValueBase
    {
      public:
      using Op = Op_;

      template <typename NumberType>
      using value_type = typename Op::template value_type<NumberType>;

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank = Op::rank;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::value;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.value,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.value,
                                                      operand.as_latex(decorator));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_values;
      }

    protected:
      // Only want this to be a base class
      explicit UnaryOpValueBase(const Op &operand)
        : operand(operand)
      {}

    private:
      const Op &operand; // TODO: Is this permitted? (temp variable?!?)
    };


    template<typename Op_>
    class UnaryOpGradientBase
    {
      public:
      using Op = Op_;

      template <typename NumberType>
      using value_type = typename Op::template gradient_type<NumberType>;

      template <typename NumberType>
      using return_type = std::vector<value_type<NumberType>>;

      static const int rank = value_type<double>::rank;

      static const enum UnaryOpCodes op_code = UnaryOpCodes::gradient;

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_ascii();
        return decorator.decorate_with_operator_ascii(naming.gradient,
                                                      operand.as_ascii(decorator));
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const auto &naming = decorator.get_naming_latex();
        return decorator.decorate_with_operator_latex(naming.gradient,
                                                      operand.as_latex(decorator));
      }

      UpdateFlags
      get_update_flags() const
      {
        return UpdateFlags::update_gradients;
      }

    protected:
      // Only want this to be a base class
      explicit UnaryOpGradientBase(const Op &operand)
        : operand(operand)
      {}

    private:
      const Op &operand; // TODO: Is this permitted? (temp variable?!?)
    };


    /* ---- Finite element spaces: Test functions and trial solutions ---- */


    /**
     * Extract the shape function values from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<Space<dim, spacedim>, UnaryOpCodes::value>
      : public UnaryOpValueBase<Space<dim, spacedim>>
    {
      using Base_t = UnaryOpValueBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

    public:
      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_value(dof_index, q_point);
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
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
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

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
    };


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

        return fe_values[this->op.extractor].value(dof_index, q_point);
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
     * Extract the shape function gradients from a finite element space.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<Space<dim, spacedim>, UnaryOpCodes::gradient>
      : public UnaryOpGradientBase<Space<dim, spacedim>>
    {
      using Base_t = UnaryOpGradientBase<Space<dim, spacedim>>;
      using typename Base_t::Op;

    public:
      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      // Return single entry
      template <typename NumberType>
      const value_type<NumberType> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 dof_index,
                 const unsigned int                 q_point) const
      {
        Assert(dof_index < fe_values.dofs_per_cell,
               ExcIndexRange(dof_index, 0, fe_values.dofs_per_cell));
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.shape_grad(dof_index, q_point);
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
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const unsigned int                 q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return_type<NumberType> out;
        out.reserve(fe_values.dofs_per_cell);

        for (const auto &dof_index : fe_values.dof_indices())
          out.emplace_back(this->operator()(fe_values, dof_index, q_point));

        return out;
      }

    protected:
      // Only want this to be a base class providing common implementation
      // for test functions / trial solutions.
      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}
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

        return fe_values[this->op.extractor].gradient(dof_index, q_point);
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


    // All test functions have the same operations as the FE space itself
    template <int dim, int spacedim, enum UnaryOpCodes OpCode>
    class UnaryOp<TestFunction<dim, spacedim>, OpCode>
      : public UnaryOp<Space<dim, spacedim>, OpCode> {
        using Op     = TestFunction<dim, spacedim>;
        using Base_t = UnaryOp<Space<dim, spacedim>, OpCode>;
        public:

          explicit UnaryOp(const Op &operand): Base_t(operand){}
      };

    // template <int dim, int spacedim, enum UnaryOpCodes OpCode>
    // class UnaryOp<SubSpaceViews::Scalar<TestFunction<dim, spacedim>>, OpCode>
    //   : public UnaryOp<SubSpaceViews::Scalar<Space<dim, spacedim>>, OpCode> {
    //     using Op     = SubSpaceViews::Scalar<TestFunction<dim, spacedim>>;
    //     using Base_t = UnaryOp<Space<dim, spacedim>, OpCode>;
    //     public:

    //       explicit UnaryOp(const Op &operand): Base_t(operand){}
    //   };


    // All trial solution have the same operations as the FE space itself
    template <int dim, int spacedim, enum UnaryOpCodes OpCode>
    class UnaryOp<TrialSolution<dim, spacedim>, OpCode>
      : public UnaryOp<Space<dim, spacedim>, OpCode> {
        using Op     = TrialSolution<dim, spacedim>;
        using Base_t = UnaryOp<Space<dim, spacedim>, OpCode>;
        public:

          explicit UnaryOp(const Op &operand): Base_t(operand){}
      };



    /* ------------ Finite element spaces: Solution fields ------------ */


    /**
     * Extract the solution values from the disretised solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<FieldSolution<dim, spacedim>, UnaryOpCodes::value>
      : public UnaryOpValueBase<FieldSolution<dim, spacedim>>
    {
      using Base_t = UnaryOpValueBase<FieldSolution<dim, spacedim>>;
      using typename Base_t::Op;

    public:
      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution values at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values.get_function_values(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution values from the disretised solution field.
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
        fe_values[this->op.extractor].get_function_values(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution gradients from the disretised solution field.
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <int dim, int spacedim>
    class UnaryOp<FieldSolution<dim, spacedim>, UnaryOpCodes::gradient>
      : public UnaryOpGradientBase<FieldSolution<dim, spacedim>>
    {
      using Base_t = UnaryOpGradientBase<FieldSolution<dim, spacedim>>;
      using typename Base_t::Op;

    public:

      template <typename NumberType> using value_type = typename Base_t::template value_type<NumberType>;
      template <typename NumberType> using return_type = typename Base_t::template return_type<NumberType>;

      explicit UnaryOp(const Op &operand)
        : Base_t(operand)
      {}

      // Return solution gradients at all quadrature points
      template <typename NumberType, typename VectorType>
      return_type<NumberType>
      operator()(const FEValuesBase<dim, spacedim> &fe_values,
                 const VectorType &                 solution) const
      {
        static_assert(
          std::is_same<NumberType, typename VectorType::value_type>::value,
          "The output type and vector value type are incompatible.");

        return_type<NumberType> out(fe_values.n_quadrature_points);
        fe_values.get_function_gradients(solution, out);
        return out;
      }
    };



    /**
     * Extract the solution values from the disretised solution field.
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
        fe_values[this->op.extractor].get_function_gradients(solution, out);
        return out;
      }
    };

  } // namespace Operators
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /* --------------- Finite element spaces: Test functions --------------- */


  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TestFunction<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TestFunction<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TestFunction<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TestFunction<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }



  /* --------------- Finite element spaces: Trial solutions --------------- */



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TrialSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::TrialSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::TrialSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TrialSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }



  /* --------------- Finite element spaces: Solution fields --------------- */



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::FieldSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::value>
  value(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = FieldSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::value>;

    return OpType(operand);
  }



  template <int dim, int spacedim>
  WeakForms::Operators::UnaryOp<WeakForms::FieldSolution<dim, spacedim>,
                                WeakForms::Operators::UnaryOpCodes::gradient>
  gradient(const WeakForms::FieldSolution<dim, spacedim> &operand)
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = FieldSolution<dim, spacedim>;
    using OpType = UnaryOp<Op, UnaryOpCodes::gradient>;

    return OpType(operand);
  }



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

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Decorator classes

  template <int dim, int spacedim>
  struct is_test_function<TestFunction<dim, spacedim>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_trial_solution<TrialSolution<dim, spacedim>> : std::true_type
  {};

  template <int dim, int spacedim>
  struct is_field_solution<FieldSolution<dim, spacedim>> : std::true_type
  {};

  // Views

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

  // template <typename SpaceType, typename std::enable_if<
  // is_test_function<SpaceType>::value || 
  // is_trial_solution<SpaceType>::value || 
  // is_field_solution<SpaceType>::value
  // >::type>
  // struct is_subspace_view<SubSpaceViews::Scalar<SpaceType>> : std::true_type
  // {};

  // template <typename SpaceType, typename std::enable_if<
  // is_test_function<SpaceType>::value || 
  // is_trial_solution<SpaceType>::value || 
  // is_field_solution<SpaceType>::value
  // >::type>
  // struct is_subspace_view<SubSpaceViews::Vector<SpaceType>> : std::true_type
  // {};

  // template <int rank, typename SpaceType, typename std::enable_if<
  // is_test_function<SpaceType>::value || 
  // is_trial_solution<SpaceType>::value || 
  // is_field_solution<SpaceType>::value
  // >::type>
  // struct is_subspace_view<SubSpaceViews::Tensor<rank,SpaceType>> : std::true_type
  // {};

  // template <int rank, typename SpaceType, typename std::enable_if<
  // is_test_function<SpaceType>::value || 
  // is_trial_solution<SpaceType>::value || 
  // is_field_solution<SpaceType>::value,SpaceType
  // >::type>
  // struct is_subspace_view<SubSpaceViews::SymmetricTensor<rank,SpaceType>> : std::true_type
  // {};

  // Unary operations

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
    Operators::UnaryOp<TestFunction<dim, spacedim>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
    Operators::UnaryOp<TrialSolution<dim, spacedim>, OpCode>> : std::true_type
  {};

  template <int dim, int spacedim, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
    Operators::UnaryOp<FieldSolution<dim, spacedim>, OpCode>> : std::true_type
  {};

  // Unary operations: Subspace views

  // template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode,
  // typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  // is_test_function<typename SubSpaceViewsType::SpaceType>::value>::type>
  // struct is_test_function<
  //   Operators::UnaryOp<SubSpaceViewsType, OpCode>> : std::true_type
  // {};

  // template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode,
  // typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  // is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
  // struct is_trial_solution<
  //   Operators::UnaryOp<SubSpaceViewsType, OpCode>> : std::true_type
  // {};

  // template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode,
  // typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  // is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>
  // struct is_field_solution<
  //   Operators::UnaryOp<SubSpaceViewsType, OpCode>> : std::true_type
  // {};

  // ----------

  // template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode>
  // struct is_test_function<
  //   Operators::UnaryOp<SubSpaceViewsType, OpCode, typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  // is_test_function<typename SubSpaceViewsType::SpaceType>::value>::type>> : std::true_type
  // {};

  // template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode>
  // struct is_trial_solution<
  //   Operators::UnaryOp<SubSpaceViewsType, OpCode, typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  // is_trial_solution<typename SubSpaceViewsType::SpaceType>::value>::type>> : std::true_type
  // {};

  // template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode>
  // struct is_field_solution<
  //   Operators::UnaryOp<SubSpaceViewsType, OpCode, typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  // is_field_solution<typename SubSpaceViewsType::SpaceType>::value>::type>> : std::true_type
  // {};

  // ----------

  template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode>
  struct is_test_function<
  typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  is_test_function<typename SubSpaceViewsType::SpaceType>::value,
  Operators::UnaryOp<SubSpaceViewsType, OpCode> 
  >::type>  : std::true_type
  {};

  template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode>
  struct is_trial_solution<
  typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  is_trial_solution<typename SubSpaceViewsType::SpaceType>::value,
  Operators::UnaryOp<SubSpaceViewsType, OpCode>
  >::type>  : std::true_type
  {};


  template <typename SubSpaceViewsType, enum Operators::UnaryOpCodes OpCode>
  struct is_field_solution<
  typename std::enable_if<is_subspace_view<SubSpaceViewsType>::value && 
  is_field_solution<typename SubSpaceViewsType::SpaceType>::value,
  Operators::UnaryOp<SubSpaceViewsType, OpCode>
  >::type>  : std::true_type
  {};

  // ============

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


  // ----------

  // template <template<class> typename SubSpaceViewsType, typename SpaceType, enum Operators::UnaryOpCodes OpCode,
  // typename std::enable_if<is_subspace_view<SubSpaceViewsType<SpaceType>>::value && 
  // is_trial_solution<SpaceType>::value>::type>
  // struct is_trial_solution<
  //   Operators::UnaryOp<SubSpaceViewsType<SpaceType>, OpCode>> : std::true_type
  // {};

  // template <template<class> typename SubSpaceViewsType, typename SpaceType, enum Operators::UnaryOpCodes OpCode,
  // typename std::enable_if<is_subspace_view<SubSpaceViewsType<SpaceType>>::value && 
  // is_field_solution<SpaceType>::value>::type>
  // struct is_field_solution<
  //   Operators::UnaryOp<SubSpaceViewsType<SpaceType>, OpCode>> : std::true_type
  // {};

} // namespace WeakForms


#endif // DOXYGEN


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_spaces_h
