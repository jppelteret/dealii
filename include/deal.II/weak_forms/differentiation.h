#ifndef dealii_weakforms_differentiation_h
#define dealii_weakforms_differentiation_h

#include <deal.II/base/config.h>

#include <deal.II/base/numbers.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>

#include <deal.II/weak_forms/cache_functors.h>

#include <tuple>
#include <type_traits>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  namespace internal
  {
    // TODO: This is replicated in energy_functor.h
    template <typename T>
    class is_scalar_type
    {
      // See has_begin_and_end() in template_constraints.h
      // and https://stackoverflow.com/a/10722840

      template <typename A>
      static constexpr auto
      test(int) -> decltype(std::declval<typename EnableIfScalar<A>::type>(),
                            std::true_type())
      {
        return true;
      }

      template <typename A>
      static std::false_type
      test(...);

    public:
      using type = decltype(test<T>(0));

      static const bool value = type::value;
    };


    template <typename T, typename U, typename = void>
    struct are_scalar_types : std::false_type
    {};


    template <typename T, typename U>
    struct are_scalar_types<
      T,
      U,
      typename std::enable_if<is_scalar_type<T>::value &&
                              is_scalar_type<U>::value>::type> : std::true_type
    {};


    // Determine types resulting from differential operations
    // of scalars, tensors and symmetric tensors.
    namespace Differentiation
    {
      template <typename T, typename U, typename = void>
      struct DiffOpResult;

      // Differentiate a scalar with respect to another scalar
      template <typename T, typename U>
      struct DiffOpResult<
        T,
        U,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = 0;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = scalar_type;

        using Op = WeakForms::ScalarCacheFunctor;
        template <int dim, int spacedim = dim>
        using function_type =
          typename Op::template function_type<scalar_type, dim, spacedim>;

        // Return unary op to functor
        template <int dim, int spacedim>
        static auto
        get_functor(const std::string &                 symbol_ascii,
                    const std::string &                 symbol_latex,
                    const function_type<dim, spacedim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim, spacedim>(
              function, UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a scalar with respect to a tensor
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        T,
        Tensor<rank_, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a scalar with respect to a symmetric tensor
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        T,
        SymmetricTensor<rank_, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type = SymmetricTensor<rank, spacedim, scalar_type>;

        using Op = SymmetricTensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a tensor with respect to a scalar
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        Tensor<rank_, spacedim, T>,
        U,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a tensor with respect to another tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        Tensor<rank_1, spacedim, T>,
        Tensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a tensor with respect to a symmetric tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        Tensor<rank_1, spacedim, T>,
        SymmetricTensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a symmetric tensor with respect to a scalar
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        SymmetricTensor<rank_, spacedim, T>,
        U,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type = SymmetricTensor<rank_, spacedim, scalar_type>;

        using Op = SymmetricTensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a symmetric tensor with respect to a tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        SymmetricTensor<rank_1, spacedim, T>,
        Tensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a symmetric tensor with respect to another symmetric
      // tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        SymmetricTensor<rank_1, spacedim, T>,
        SymmetricTensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type = SymmetricTensor<rank, spacedim, scalar_type>;

        using Op = SymmetricTensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return unary op to functor
        template <int dim, int /*spacedim*/>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function)
        {
          return get_operand(symbol_ascii, symbol_latex)
            .template value<scalar_type, dim>(function,
                                              UpdateFlags::update_default);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Specialization for differentiating a scalar or tensor with respect
      // to a tuple of fields.
      // This is only intended for use in a very narrow context, so we only
      // define the result types for the operation.
      //
      // This builds up into a tuple that can be represented as:
      // [ df/dx_1 , df/dx_2 , ... , df/dx_n ]
      template <typename T, typename... Us>
      struct DiffOpResult<T, std::tuple<Us...>, void>
      {
        // The result type
        using type = std::tuple<typename DiffOpResult<T, Us>::type...>;
      };

      // Specialization for differentiating a tuple of scalars or tensors
      // with respect to a tuple of fields.
      // This is only intended for use in a very narrow context, so we only
      // define the result types for the operation.
      //
      // This builds up into a nested tuple with the following
      // structure/grouping:
      // [ [ d^2f/dx_1.dx_1 , d^2f/dx_1.dx_2 , ... , d^2f/dx_1.dx_n ] ,
      //   [ d^2f/dx_2.dx_1 , d^2f/dx_2.dx_2 , ... , d^2f/dx_2.dx_n ] ,
      //   [                   ...                                  ] ,
      //   [ d^2f/dx_n.dx_1 , d^2f/dx_n.dx_2 , ... , d^2f/dx_n.dx_n ] ]
      //
      // So the outer tuple holds the "row elements", and the inner tuple
      // the "column elements" for each row.
      template <typename... Ts, typename... Us>
      struct DiffOpResult<std::tuple<Ts...>, std::tuple<Us...>, void>
      {
        // The result type
        using type =
          std::tuple<typename DiffOpResult<Ts, std::tuple<Us...>>::type...>;
      };

    } // namespace Differentiation
  }   // namespace internal
} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_differentiation_h
