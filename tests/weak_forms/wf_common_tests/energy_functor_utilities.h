
#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/weak_forms/energy_functor.h>
#include <deal.II/weak_forms/symbolic_decorations.h>


DEAL_II_NAMESPACE_OPEN

namespace WeakForms
{
  namespace Operators
  {
    template <typename... UnaryOpsSubSpaceFieldSolution>
    using OpHelper_t = internal::UnaryOpsSubSpaceFieldSolutionHelper<
      UnaryOpsSubSpaceFieldSolution...>;

    // End point
    template <std::size_t I = 0, typename... UnaryOpType>
    static typename std::enable_if<I == sizeof...(UnaryOpType), void>::type
    unpack_print_field_args_and_extractors(
      const typename OpHelper_t<UnaryOpType...>::field_args_t &field_args,
      const typename OpHelper_t<UnaryOpType...>::field_extractors_t
        &                        field_extractors,
      const SymbolicDecorations &decorator)
    {
      (void)field_args;
      (void)field_extractors;
    }


    template <std::size_t I = 0, typename... UnaryOpType>
      static typename std::enable_if <
      I<sizeof...(UnaryOpType), void>::type
      unpack_print_field_args_and_extractors(
        const typename OpHelper_t<UnaryOpType...>::field_args_t &field_args,
        const typename OpHelper_t<UnaryOpType...>::field_extractors_t
          &                        field_extractors,
        const SymbolicDecorations &decorator)
    {
      deallog << "Field index  " << dealii::Utilities::to_string(I) << ": "
              << std::get<I>(field_args).as_ascii(decorator) << " -> "
              << std::get<I>(field_extractors).get_name() << std::endl;

      unpack_print_field_args_and_extractors<I + 1, UnaryOpType...>(
        field_args, field_extractors, decorator);
    }

    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename... UnaryOpsSubSpaceFieldSolution>
    void
    print_field_args_and_extractors(
      const UnaryOp<
        WeakForms::EnergyFunctor<UnaryOpsSubSpaceFieldSolution...>,
        UnaryOpCodes::value,
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
        ADNumberType,
        WeakForms::internal::DimPack<dim, spacedim>> &energy_functor,
      const WeakForms::SymbolicDecorations &          decorator)
    {
      deallog
        << "Number of components: "
        << dealii::Utilities::to_string(
             OpHelper_t<UnaryOpsSubSpaceFieldSolution...>::get_n_components())
        << std::endl;

      unpack_print_field_args_and_extractors<0,
                                             UnaryOpsSubSpaceFieldSolution...>(
        energy_functor.get_field_args(),
        energy_functor.get_field_extractors(),
        decorator);
    }

  } // namespace Operators
} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE
