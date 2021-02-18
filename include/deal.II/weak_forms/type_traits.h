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

#ifndef dealii_weakforms_type_traits_h
#define dealii_weakforms_type_traits_h

#include <deal.II/base/config.h>

#include <type_traits>


DEAL_II_NAMESPACE_OPEN


namespace WeakForms
{
  template <typename T>
  struct is_test_function : std::false_type
  {};

  template <typename T>
  struct is_trial_solution : std::false_type
  {};

  template <typename T, typename U = void>
  struct is_test_function_or_trial_solution : std::false_type
  {};

  template <typename T>
  struct is_test_function_or_trial_solution<
    T,
    typename std::enable_if<is_test_function<T>::value ||
                            is_trial_solution<T>::value>::type> : std::true_type
  {};

  template <typename T>
  struct is_field_solution : std::false_type
  {};

  template <typename T>
  struct is_subspace_view : std::false_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_cache_functor : std::false_type
  {};


  // TODO: Add test for this
  template <typename T>
  struct is_ad_functor : std::false_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_sd_functor : std::false_type
  {};


  // TODO: Add test for this
  template <typename T, typename U = void>
  struct evaluates_with_scratch_data : std::false_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct evaluates_with_scratch_data<
    T,
    typename std::enable_if<is_field_solution<T>::value ||
                            is_cache_functor<T>::value
                            // || is_ad_functor<T>::value
                            // || is_sd_functor<T>::value
                            >::type> : std::true_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_bilinear_form : std::false_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_linear_form : std::false_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_self_linearizing_form : std::false_type
  {};

  // TODO: Add this to pre-existing test
  template <typename T>
  struct is_symbolic_volume_integral : std::false_type
  {};

  // TODO: Add this to pre-existing test
  template <typename T>
  struct is_symbolic_boundary_integral : std::false_type
  {};

  // TODO: Add this to pre-existing test
  template <typename T>
  struct is_symbolic_interface_integral : std::false_type
  {};

  template <typename T, typename U = void>
  struct is_symbolic_integral : std::false_type
  {};

  template <typename T>
  struct is_symbolic_integral<
    T,
    typename std::enable_if<is_symbolic_volume_integral<T>::value ||
                            is_symbolic_boundary_integral<T>::value ||
                            is_symbolic_interface_integral<T>::value>::type>
    : std::true_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_symbolic_op : std::false_type
  {};

  // TODO: Add test for this
  template <typename T, typename U = void>
  struct is_unary_op : std::false_type
  {};

  template <typename T>
  struct is_unary_op<T, typename std::enable_if<is_symbolic_op<T>::value>::type>
    : std::true_type
  {};

  // TODO: Add test for this
  template <typename T>
  struct is_binary_op : std::false_type
  {};

} // namespace WeakForms


DEAL_II_NAMESPACE_CLOSE

#endif // dealii_weakforms_type_traits_h
