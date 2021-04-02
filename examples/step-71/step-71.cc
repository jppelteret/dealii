/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Jean-Paul Pelteret, 2020
 */


// We start by including all the necessary deal.II header files and some C++
// related ones.
// This first header will give us access to a data structure that will allow
// us to store arbitrary data within it.
#include <deal.II/algorithms/general_data_storage.h>

// Next come some core classes, including one that provides an implementation
// for time-stepping.
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

// These headers define some useful coordinate transformations and kinematic
// relationships that are often found in nonlinear elasticity.
#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

// The following two headers provide all of the functionality that we need
// to perform automatic differentiation, and use the symbolic computer algebra
// system that deal.II can utilize. The headers of the all automatic
// differentiation and symbolic differentiation wrapper classes, and any
// ancillary data structures that are required, are all collected inside these
// unifying headers.
#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

// Including this header allows us the capability to write output to a
// file stream.
#include <fstream>


// As per usual, the entire tutorial program is defined within its own unique
// namespace.
namespace Step71
{
  // Not only will we expose the deal.II namespace within this tutorial, but
  // also that encapsulating deal.II's differentiation classes. This will save
  // us some effort when referencing these classes later.
  using namespace dealii;
  using namespace dealii::Differentiation;

  // @sect3{An introductory example: The fundamentals of automatic and symbolic differentiation}
  //
  // Automatic and symbolic differentiation have some magical and mystical
  // qualities. Although their use in a project can be beneficial for a
  // multitude of reasons, the barrier to understanding how to use these
  // frameworks or how they can be leveraged may exceed the patience of
  // the developer that is trying to (reliably) integrate them into their work.
  //
  // Although it is the wish of the author to successfully illustrate how these
  // tools can be intergrated into workflows for finite element modelling, it
  // might be best to first take a step back and start right from the basics.
  // So to start off with, we'll first have a look at differentiating a "simple"
  // mathematical function using both frameworks, so that the fundamental
  // operations (both their sequence and function) can be firmly established and
  // understood with minimal complication. In the second part of this tutorial
  // we will put these fundamentals into practice and build on them further.
  //
  // Accompanying the description of the algorithmic steps to use the frameworks
  // will be a simplified view as to what they *might* be doing in the
  // background. This description will be very much one designed to aid
  // understanding, and the reader is encouraged to view the @ref auto_symb_diff
  // module documentation for a far more formal description into how these tools
  // actually work.
  namespace SimpleExample
  {
    // @sect4{Analytical function}

    // In order to convince the reader that these tools are indeed useful in
    // practice, let us choose a function that's not too difficult to compute
    // the analytical solution to its derivatives by hand. Its just sufficiently
    // complicated to make you think about whether or not you truly want to go
    // through with this exercise, and might also make you question whether you
    // are completely sure that your calculations and implementation for its
    // derivatives are correct.
    //
    // We choose the two variable trigonometric function
    // $f(x,y) = \cos(\frac{y}{x})$ for this purpose. Notice that this function
    // is templated on the number type. This is done because we can often (but
    // not always) use the auto-differentiable and symbolic types as drop-in
    // replacements for real or complex valued functions that perform some
    // elementary calculations, such as evaluate a function value. We will
    // exploit that property and make sure that we need only define our function
    // once, and then it can be re-used in whichever context we wish to perform
    // differential operations on it.
    template <typename NumberType>
    NumberType f(const NumberType &x, const NumberType &y)
    {
      return std::cos(y / x);
    }

    // Rather than reveal this functions derivatives immediately, we'll
    // forward declare functions that return the them and rather define them
    // later. As implied by the function names, they respectively return
    // the derivatives
    //
    // $\frac{df(x,y)}{dx}$,
    double df_dx(const double &x, const double &y);

    // $\frac{df(x,y)}{dy}$,
    double df_dy(const double &x, const double &y);

    // $\frac{d^{2}f(x,y)}{dx^{2}}$,
    double d2f_dx_dx(const double &x, const double &y);

    // $\frac{d^{2}f(x,y)}{dx dy}$,
    double d2f_dx_dy(const double &x, const double &y);

    // $\frac{d^{2}f(x,y)}{dy dx}$,
    double d2f_dy_dx(const double &x, const double &y);

    // and, lastly, $\frac{d^{2}f(x,y)}{dy^{2}}$.
    double d2f_dy_dy(const double &x, const double &y);


    // @sect4{Computing derivatives using automatic differentiation}

    // To begin, we'll use AD as the tool to automatically
    // compute derivatives for us. We will evaluate the function with the
    // arguments `x` and `y`, and expect the resulting value and all of the
    // derivatives to match to within the given tolerance.
    void run_and_verify_ad(const double &x,
                           const double &y,
                           const double  tol = 1e-12)
    {
      // Our function $f(x,y)$ is a scalar-valued function, with arguments that
      // represent the typical input variables that one comes across in
      // algebraic calculations or tensor calculus. For this reason, the
      // Differentiation::AD::ScalarFunction class is the appropriate wrapper
      // class to use to do the computations that we require. (As a point of
      // comparison, if the function arguments represented finite element cell
      // degrees-of-freedom, we'd want to treat them differently.) The spatial
      // dimension of the problem is irrelevant since we have no vector- or
      // tensor-valued arguments to accommodate, so the `dim` template argument
      // is arbitrarily assigned a value of 1. The second template argument
      // stipulates which AD framework will be used, and what the underlying
      // number type provided by this framework is to be employed. This number
      // type influences the maximum order of the differential operation, and
      // the underlying algorithms that are used to compute them. Given its
      // template nature, this choice is a compile-time decision because many
      // (but not all) of the AD libraries exploit
      // compile-time meta-programming to implement these special number types
      // in an efficient manner. The third template parameter states what the
      // result type is; in our case, we're working with `double`s.
      constexpr unsigned int    dim        = 1;
      constexpr AD::NumberTypes ADTypeCode = AD::NumberTypes::sacado_dfad_dfad;
      using ADHelper = AD::ScalarFunction<dim, ADTypeCode, double>;

      // It is necessary that we pre-register with our @p ADHelper class how many
      // arguments (what we will call "independent variables") the function
      // $f(x,y)$ has. Those arguments are `x` and `y`, so obviously there
      // are two of them.
      constexpr unsigned int n_independent_variables = 2;

      // We now have sufficient information to create and initialize an
      // instance of the helper class. We can also get the concrete
      // number type that will be used in all subsequent calculations.
      // This is useful, because we can write everything from here on by
      // referencing this type, and if we ever want to change the framework
      // used, or number type (e.g., if we need more differential operations)
      // then we need only adjust the `ADTypeCode` template parameter.
      ADHelper ad_helper(n_independent_variables);
      using ADNumberType = typename ADHelper::ad_type;

      // The next step is to register the numerical values of the independent
      // variables with the helper class. This is done because the function
      // and its derivatives will be evaluated for exactly these arguments.
      // Since we register them in the order `{x,y}`, the variable `x` will
      // be assigned component number `0`, and `y` will be component `1`
      // -- a detail that will be used in the next few lines.
      ad_helper.register_independent_variables({x, y});

      // We now ask for the helper class to give to us the independent variables
      // with their auto-differentiable representation. These are termed
      // "sensitive variables", because from this point on any operations that
      // we do with the components `independent_variables_ad` are tracked and
      // recorded by the AD framework, and will be considered
      // when we ask for the derivatives of something that they're used to
      // compute. What the helper returns is a `vector` of auto-differentiable
      // numbers, but we can be sure that the zeroth element represents `x`
      // and the first element `y`. Just to make completely sure that there's
      // no ambiguity of what number type these variables are, we suffix all of
      // the auto-differentiable variables with `ad`.
      const std::vector<ADNumberType> independent_variables_ad =
        ad_helper.get_sensitive_variables();
      const ADNumberType &x_ad = independent_variables_ad[0];
      const ADNumberType &y_ad = independent_variables_ad[1];

      // We can immediately pass in our sensitive representation of the
      // independent variables to our templated function that computes
      // $f(x,y)$.
      // This also returns an auto-differentiable number.
      const ADNumberType f_ad = f(x_ad, y_ad);

      // So now the natural question to ask is how all of the derivatives that
      // you're wanting are determined from `f_ad`. What is so special about
      // this `ADNumberType` that gives it the ability to magically return
      // derivatives?
      //
      // In essence, how this *could* be done is the following:
      // This special number can be viewed as a data structure that stores the
      // function value, and the prescribed number of derivatives. For a
      // once-differentiable number expecting two arguments, it might look like
      // this:
      //
      // @code
      // class ADNumberType
      // {
      //   double f;     // Function value f(x,y)
      //   double df[2]; // Array of function derivatives
      //                 // [df(x,y)/dx, df(x,y)/dx]
      // };
      // @endcode
      //
      // For our independent variable `x`, the starting value of `x.f` would
      // simply be its assigned value (i.e., the real value of that this
      // variable represents). The derivative `x.df[0]` would be initialized to
      // `1`, since `x` is the zeroth independent variable and
      // $\frac{d(x)}{dx} = 1$. The derivative `x.df[1]` would be initialized to
      // zero, since the first independent variable is `y` and
      // $\frac{d(x)}{dy} = 0$.
      //
      // For the function derivatives to be meaningful, we must assume that not
      // only is this function differentiable in an analytical sense, but that
      // its also differentiable at the evaluation point `x,y`.
      // We can exploit both of these assumptions: when we use this number type
      // in mathematical operations, the AD framework *could*
      // overload the operations (e.g., `%operator+()`, `%operator*()` as well
      // as `%sin()`, `%exp()`, etc.) such that the returned result has the
      // expect value. At the same time, it would then compute the derivatives
      // through the knowledge of exactly what function is being overloaded and
      // rigorous application of the chain-rule. So, the `%sin()` function
      // *might* be defined as follows:
      //
      // @code
      // ADNumberType sin(const ADNumberType &a)
      // {
      //   ADNumberType output;
      //
      //   // For the input argument "a", "a.f" is simply its value.
      //   output.f = sin(a.f);
      //
      //   // We know that the derivative of sin(a) is cos(a), but we need
      //   // to also consider the chain rule and that the input argument
      //   // `a` is also differentiable with respect to the original
      //   // independent variables `x` and `y`. So `a.df[0]` and `a.df[1]`
      //   // respectively represent the partial derivatives of `a` with
      //   // respect to its inputs `x` and `y`.
      //   output.df[0] = cos(a.f)*a.df[0];
      //   output.df[1] = cos(a.f)*a.df[1];
      //
      //   return output;
      // };
      // @endcode
      //
      // So it is now clear that with the above representation the
      // `ADNumberType` is carrying around some extra data that represents the
      // various derivatives of differentiable functions with respect to the
      // original (sensitive) independent variables. It should therefore be
      // noted that there is computational overhead associated with using them
      // (as we compute extra functions when doing derivative computations) as
      // well as memory overhead in storing these results. So the prescribed
      // number of levels of differential operations should ideally be kept to a
      // minimum to limit computational cost. We could, for instance, have
      // computed the first derivatives ourself and then have used the
      // Differentiation::AD::VectorFunction helper class to determine the
      // gradient of the collection of dependent functions, which would be the
      // second derivatives of the original scalar function.
      //
      // It is also worth noting that because the chain rule is indiscriminately
      // applied and we only see the beginning and end-points of the calculation
      // `{x,y}` $\rightarrow$ `f(x,y)`, we will only ever be able to query
      // the total derivatives of `f`; the partial derivatives (`a.df[0]` and
      // `a.df[1]` in the above example) are intermediate values and are hidden
      // from us.

      // Okay, since we now at least have some idea as to exactly what `f_ad`
      // represents and what is encoded within it, let's put all of that to
      // some actual use. The gain access to those hidden derivative results,
      // we register the final result with the helper class. After this point,
      // we can no longer change the value of `f_ad` and have those changes
      // reflected in the results returned by the helper class.
      ad_helper.register_dependent_variable(f_ad);

      // The next step is to extract the derivatives (specifically, the function
      // gradient and Hessian). To do so we first create some temporary data
      // structures (with the result type `double`) to store the derivatives
      // (noting that all derivatives are returned at once, and not
      // individually)...
      Vector<double>     Df(ad_helper.n_dependent_variables());
      FullMatrix<double> D2f(ad_helper.n_dependent_variables(),
                             ad_helper.n_independent_variables());

      // ... and we then request that the helper class compute these
      // derivatives, and the function value itself. And that's it. We have
      // everything that we were aiming to get.
      const double computed_f = ad_helper.compute_value();
      ad_helper.compute_gradient(Df);
      ad_helper.compute_hessian(D2f);

      // We can convince ourselves that the AD framework is
      // correct by comparing it to the analytical solution. However, if you're
      // like the author, you'll be doing the opposite and will rather verify
      // that your implementation of the analytical solution is correct!
      AssertThrow(std::abs(f(x, y) - computed_f) < tol,
                  ExcMessage(std::string("Incorrect value computed for f. ") +
                             std::string("Hand-calculated value: ") +
                             Utilities::to_string(f(x, y)) +
                             std::string(" ; ") +
                             std::string("Value computed by AD: ") +
                             Utilities::to_string(computed_f)));

      // Because we know the ordering of the independent variables, we know
      // which component of the gradient relates to which derivative...
      const double computed_df_dx = Df[0];
      const double computed_df_dy = Df[1];

      AssertThrow(std::abs(df_dx(x, y) - computed_df_dx) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for df/dx. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(df_dx(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_df_dx)));
      AssertThrow(std::abs(df_dy(x, y) - computed_df_dy) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for df/dy. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(df_dy(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_df_dy)));

      // ... and similar for the Hessian.
      const double computed_d2f_dx_dx = D2f[0][0];
      const double computed_d2f_dx_dy = D2f[0][1];
      const double computed_d2f_dy_dx = D2f[1][0];
      const double computed_d2f_dy_dy = D2f[1][1];

      AssertThrow(std::abs(d2f_dx_dx(x, y) - computed_d2f_dx_dx) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dx_dx. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dx_dx(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_d2f_dx_dx)));
      AssertThrow(std::abs(d2f_dx_dy(x, y) - computed_d2f_dx_dy) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dx_dy. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dx_dy(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_d2f_dx_dy)));
      AssertThrow(std::abs(d2f_dy_dx(x, y) - computed_d2f_dy_dx) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dy_dx. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dy_dx(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_d2f_dy_dx)));
      AssertThrow(std::abs(d2f_dy_dy(x, y) - computed_d2f_dy_dy) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dy_dy. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dy_dy(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_d2f_dy_dy)));
    }

    // That's pretty great. There wasn't too much work involved in computing
    // second-order derivatives of this trigonometric function.

    // @sect4{Hand-calculated derivatives of the analytical solution}

    // Since we now know how much "implementation effort" it takes to have the
    // AD framework compute those derivatives for us, let's
    // compare that to the same computed by hand and implemented in several
    // stand-alone methods.

    // Here are the two first derivatives of $f(x,y) = \cos(\frac{y}{x})$:
    //
    // $\frac{df(x,y)}{dx} = \frac{y}{x^2} \sin(\frac{y}{x})$
    double df_dx(const double &x, const double &y)
    {
      return y * std::sin(y / x) / (x * x);
    }

    // $\frac{df(x,y)}{dx} = -\frac{1}{x} \sin(\frac{y}{x})$
    double df_dy(const double &x, const double &y)
    {
      return -std::sin(y / x) / x;
    }

    // And here are the four second derivatives of $f(x,y)$:
    //
    // $\frac{d^{2}f(x,y)}{dx^{2}} = -\frac{y}{x^4} (2x \sin(\frac{y}{x}) + y
    // \cos(\frac{y}{x}))$
    double d2f_dx_dx(const double &x, const double &y)
    {
      return -y * (2 * x * std::sin(y / x) + y * std::cos(y / x)) /
             (x * x * x * x);
    }

    // $\frac{d^{2}f(x,y)}{dx dy} = \frac{1}{x^3} (x \sin(\frac{y}{x}) + y
    // \cos(\frac{y}{x}))$
    double d2f_dx_dy(const double &x, const double &y)
    {
      return (x * std::sin(y / x) + y * std::cos(y / x)) / (x * x * x);
    }

    // $\frac{d^{2}f(x,y)}{dy dx} = \frac{1}{x^3} (x \sin(\frac{y}{x}) + y
    // \cos(\frac{y}{x}))$ (as expected, on the basis of [Schwarz's
    // theorem](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives))
    double d2f_dy_dx(const double &x, const double &y)
    {
      return (x * std::sin(y / x) + y * std::cos(y / x)) / (x * x * x);
    }

    // $\frac{d^{2}f(x,y)}{dy^{2}} = -\frac{1}{x^2} \cos(\frac{y}{x})$
    double d2f_dy_dy(const double &x, const double &y)
    {
      return -(std::cos(y / x)) / (x * x);
    }

    // Hmm... there's a lot of places in the above that we could have introduced
    // an error in the above, especially when it comes to employing the chain
    // rule. Although they're no silver bullet, at the very least these
    // AD frameworks can serve as a verification tool to make
    // sure that we haven't made any errors (either by calculation or by
    // implementation) that would negatively affect our results. Another benefit
    // is that if we extend our original function to have more input arguments,
    // then there is only a trivial adjustment required to have the
    // AD framework accommodate those new variables.


    // @sect4{Computing derivatives using symbolic differentiation}

    // We'll now repeat the same exercise using symbolic differentiation. The
    // term "symbolic differentiation" is a little bit misleading because
    // differentiation is just one tool that the Computer Algebra System (CAS)
    // (i.e., the symbolic framework) provides. Nevertheless, in the context
    // of finite element modelling and applications it is the most common use
    // of a CAS and will therefore be the one that we'll focus on.
    // Once more we'll supply the argument values `x` and `y` with which to
    // evaluate our function $f(x,y) = \cos(\frac{y}{x})$ and its derivatives,
    // and a tolerance with which to test the correctness of the returned
    // results.
    void run_and_verify_sd(const double &x,
                           const double &y,
                           const double  tol = 1e-12)
    {
      // The first step that we need to take is to form to symbolic variables
      // that represent the function arguments that we wish to differentiate
      // with respect to. Again, these will be the independent variables for
      // our problem and as such are, in some sense, primitive variables that
      // have no dependencies on any other variable. We create these types of
      // (independent) variables by initializing a symbolic type
      // Differentiation::SD::Expression, which is a wrapper to a set of classes
      // used by the symbolic framework, with a unique identifier. On this
      // occasion it makes sense that this identifier, a `std::string`, be
      // simply `"x"` for the $x$ argument, and likewise `"y"` for the $y$
      // argument to the dependent function. Like before, we'll suffix symbolic
      // variable names with `sd` so that we can clearly see which variables are
      // symbolic (as opposed to numeric) in nature.
      const SD::Expression x_sd("x");
      const SD::Expression y_sd("y");

      // Using the templated function that computes $f(x,y)$, we can pass
      // these independent variables as arguments to the function. The returned
      // result will be another symbolic type that represents sequence of
      // operations used to compute $\cos(\frac{y}{x})$.
      const SD::Expression f_sd = f(x_sd, y_sd);

      // At this point it is legitimate to print out the expression `f_sd`, and
      // if we did so
      // @code
      // std::cout << "f(x,y) = " << f_sd << std::endl;
      // @endcode
      // we would see `f(x,y) = cos(y/x)` printed to the console.
      //
      // You might notice that we've constructed our symbolic function `f_sd`
      // with no context as to how we might want to use it. This is one of the
      // key points that makes symbolic frameworks (the CAS) different to
      // automatic differentiation frameworks. Each of the variables `x_sd` and
      // `y_sd`, and even the composite dependent function `f_sd`, are in some
      // sense respectively "placeholders" for numerical values and a
      // composition of operations. In fact, the individual components that are
      // used to compose the function are also placeholders. The sequence of
      // operations are encoded into in a tree-like data structure (conceptually
      // simlar to a [abstract syntax
      // tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree)).
      //
      // Once we form these data structures we can defer any operations that we
      // might want to do with them until some later time. Each of these
      // placeholders represent something, but we have the opportunity to define
      // or redefine what they represent at any convenient point in time. So for
      // this particular problem it makes sense that somehow want to associate
      // "x" and "y" with *some* numerical value (with type yet to be
      // determined), but we could conceptually (and if it made sense) assign
      // the ratio "y/x" a value instead of the variables "x" and "y"
      // individually. We could also associate with "x" or "y" some other
      // symbolic function `g(a,b)`. Any of these operations involves
      // manipulating the recorded tree of operations, and substituting the
      // salient nodes on the tree (and that nodes' subtree) with something
      // else.
      //
      // This capability makes framework is entirely generic.
      // The types of operations that, in the context of finite element
      // simulations, we would typically perform with our symbolic types are
      // function composition, differentiation, substitution (partial or
      // complete) and evaluation (i.e., conversion of the symbolic type to its
      // numerical counterpart). But should you need it, a CAS is often capable
      // of more than just this.

      // To compute the symbolic representation of the first derivatives of
      // the dependent function with respect to its individual independent
      // variables, we use the Differentiation::SD::Expression::differentiate()
      // function with the independent variable given as its argument. Each call
      // will cause the CAS to parse the tree of operations that compose `f_sd`
      // and differentiate each node of the expression tree with respect to the
      // given symbolic argument.
      const SD::Expression df_dx_sd = f_sd.differentiate(x_sd);
      const SD::Expression df_dy_sd = f_sd.differentiate(y_sd);

      // To compute the symbolic representation of the second derivatives, we
      // simply differentiate the first derivatives with respect to the
      // independent variables. So to compute a higher order derivative, we
      // first need to compute the lower order derivative.
      // (As the return type of the call to `differentiate()` is an expression,
      // we could in principal execute double differentiation directly from the
      // scalar by chaining two calls together. But this is unnecessary in this
      // particular case, since we have the intermediate results at hand.)
      const SD::Expression d2f_dx_dx_sd = df_dx_sd.differentiate(x_sd);
      const SD::Expression d2f_dx_dy_sd = df_dx_sd.differentiate(y_sd);
      const SD::Expression d2f_dy_dx_sd = df_dy_sd.differentiate(x_sd);
      const SD::Expression d2f_dy_dy_sd = df_dy_sd.differentiate(y_sd);
      // Printing the expressions for the first and second derivatives, as
      // computed by the CAS, using the statements
      // @code
      // std::cout << "df_dx_sd: " << df_dx_sd << std::endl;
      // std::cout << "df_dy_sd: " << df_dy_sd << std::endl;
      // std::cout << "d2f_dx_dx_sd: " << d2f_dx_dx_sd << std::endl;
      // std::cout << "d2f_dx_dy_sd: " << d2f_dx_dy_sd << std::endl;
      // std::cout << "d2f_dy_dx_sd: " << d2f_dy_dx_sd << std::endl;
      // std::cout << "d2f_dy_dy_sd: " << d2f_dy_dy_sd << std::endl;
      // @endcode
      // renders the following output:
      // @code{.sh}
      // df_dx_sd: y*sin(y/x)/x**2
      // df_dy_sd: -sin(y/x)/x
      // d2f_dx_dx_sd: -y**2*cos(y/x)/x**4 - 2*y*sin(y/x)/x**3
      // d2f_dx_dy_sd: sin(y/x)/x**2 + y*cos(y/x)/x**3
      // d2f_dy_dx_sd: sin(y/x)/x**2 + y*cos(y/x)/x**3
      // d2f_dy_dy_sd: -cos(y/x)/x**2
      // @endcode
      // This compares favorably to the analytical expressions for these
      // derivatives that were presented earlier.

      // Now that we have formed the symbolic expressions for the function and
      // its derivatives, we want to evalute them for the numeric values for
      // the main function arguments `x` and `y`. To do this we construct a
      // substitution map, which maps the symbolic values to their numerical
      // counterparts.
      const SD::types::substitution_map substitution_map =
        SD::make_substitution_map(std::pair<SD::Expression, double>{x_sd, x},
                                  std::pair<SD::Expression, double>{y_sd, y});

      // The last step in the process is to convert all symbolic variables and
      // operations into numerical values, and produce the numerical result of
      // this operation. To do this we combine the substitution map with the
      // symbolic variable in a step called "substitution".
      //
      // Once we pass this substitution map to the CAS, it will
      // substitute each instance of the symbolic variable with its numerical
      // counterpart and then propogate these results up the operation tree,
      // simplifying each node on the tree if possible. If the tree is
      // reduced to a single value (i.e., we have substituted all of the
      // independent variables with their numerical counterpart) then the
      // evaluation is complete.
      //
      // Due to the strongly-typed nature of C++, we need to instruct the CAS to
      // convert its representation of the result into an intrinsic data type
      // (in this case a `double`). This is the "evaluation" step, and through
      // the template type we define the return type of this process.
      // Conveniently, these two steps can be done at once if we are certain
      // that we've performed a full substitution.
      const double computed_f =
        f_sd.substitute_and_evaluate<double>(substitution_map);

      AssertThrow(std::abs(f(x, y) - computed_f) < tol,
                  ExcMessage(std::string("Incorrect value computed for f. ") +
                             std::string("Hand-calculated value: ") +
                             Utilities::to_string(f(x, y)) +
                             std::string(" ; ") +
                             std::string("Value computed by AD: ") +
                             Utilities::to_string(computed_f)));

      // We can do the same for the first derivatives...
      const double computed_df_dx =
        df_dx_sd.substitute_and_evaluate<double>(substitution_map);
      const double computed_df_dy =
        df_dy_sd.substitute_and_evaluate<double>(substitution_map);

      AssertThrow(std::abs(df_dx(x, y) - computed_df_dx) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for df/dx. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(df_dx(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_df_dx)));
      AssertThrow(std::abs(df_dy(x, y) - computed_df_dy) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for df/dy. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(df_dy(x, y)) + std::string(" ; ") +
                    std::string("Value computed by AD: ") +
                    Utilities::to_string(computed_df_dy)));

      // ... and the second derivatives.
      // Notice that we can reuse the same substitution map for each of these
      // operations because we wish to evaluate all of these functions for the
      // same values of `x` and `y`. Modifying the values in the substitution
      // map renders the result of same symbolic expression evaluated with
      // different values being assigned to the independent variables.
      // We could also happily have each variable represent a real value in
      // one pass, and a complex value in the next.
      const double computed_d2f_dx_dx =
        d2f_dx_dx_sd.substitute_and_evaluate<double>(substitution_map);
      const double computed_d2f_dx_dy =
        d2f_dx_dy_sd.substitute_and_evaluate<double>(substitution_map);
      const double computed_d2f_dy_dx =
        d2f_dy_dx_sd.substitute_and_evaluate<double>(substitution_map);
      const double computed_d2f_dy_dy =
        d2f_dy_dy_sd.substitute_and_evaluate<double>(substitution_map);

      AssertThrow(std::abs(d2f_dx_dx(x, y) - computed_d2f_dx_dx) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dx_dx. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dx_dx(x, y)) + std::string(" ; ") +
                    std::string("Value computed by SD: ") +
                    Utilities::to_string(computed_d2f_dx_dx)));
      AssertThrow(std::abs(d2f_dx_dy(x, y) - computed_d2f_dx_dy) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dx_dy. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dx_dy(x, y)) + std::string(" ; ") +
                    std::string("Value computed by SD: ") +
                    Utilities::to_string(computed_d2f_dx_dy)));
      AssertThrow(std::abs(d2f_dy_dx(x, y) - computed_d2f_dy_dx) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dy_dx. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dy_dx(x, y)) + std::string(" ; ") +
                    std::string("Value computed by SD: ") +
                    Utilities::to_string(computed_d2f_dy_dx)));
      AssertThrow(std::abs(d2f_dy_dy(x, y) - computed_d2f_dy_dy) < tol,
                  ExcMessage(
                    std::string("Incorrect value computed for d2f/dy_dy. ") +
                    std::string("Hand-calculated value: ") +
                    Utilities::to_string(d2f_dy_dy(x, y)) + std::string(" ; ") +
                    std::string("Value computed by SD: ") +
                    Utilities::to_string(computed_d2f_dy_dy)));
    }

  } // namespace SimpleExample


  // @sect3{A more complex example: Using automatic and symbolic differentiation to compute derivatives at continuum points}
  //
  // Now that we've introduced the principles behind automatic and symbolic
  // differentiation, we'll put them into action by formulating a coupled
  // magneto-mechanical constitutive law.

  namespace CoupledConstitutiveLaws
  {
    // @sect4{Constitutive parameters}

    // Values for all parameters (constitutive + rheological) taken from
    // @cite Pelteret2018a
    class ConstitutiveParameters : public ParameterAcceptor
    {
    public:
      ConstitutiveParameters();

      // The first four constitutive parameters respectively represent
      // - the elastic shear modulus,
      // - the elastic shear modulus at magnetic saturation,
      // - the saturation magnetic field strength for the elastic shear
      //   modulus, and
      // - the Poisson ratio.
      double mu_e       = 30.0e3;
      double mu_e_inf   = 250.0e3;
      double mu_e_h_sat = 212.2e3;
      double nu_e       = 0.49;

      // The next four are parameters for
      // - the viscoelastic shear modulus
      // - the viscoelastic shear modulus at magnetic saturation
      // - the saturation magnetic field strength for the viscoelastic
      //   shear modulus, and
      // - the characteristic relaxation time.
      double mu_v       = 20.0e3;
      double mu_v_inf   = 35.0e3;
      double mu_v_h_sat = 92.84e3;
      double tau_v      = 0.6;

      // The last parameter is the relative magnetic permeability.
      double mu_r = 6.0;

      bool initialized = false;
    };

    ConstitutiveParameters::ConstitutiveParameters()
      : ParameterAcceptor("/Coupled Constitutive Laws/Constitutive Parameters/")
    {
      add_parameter("Elastic shear modulus", mu_e);
      add_parameter("Elastic shear modulus at magnetic saturation", mu_e_inf);
      add_parameter(
        "Saturation magnetic field strength for elastic shear modulus",
        mu_e_h_sat);
      add_parameter("Poisson ratio", nu_e);

      add_parameter("Viscoelastic shear modulus", mu_v);
      add_parameter("Viscoelastic shear modulus at magnetic saturation",
                    mu_v_inf);
      add_parameter(
        "Saturation magnetic field strength for viscoelastic shear modulus",
        mu_v_h_sat);
      add_parameter("Characteristic relaxation time", tau_v);

      add_parameter("Relative magnetic permeability", mu_r);

      parse_parameters_call_back.connect([&]() -> void { initialized = true; });
    }

    // @sect4{Constitutive laws: Base class}
    template <int dim>
    class Coupled_Magnetomechanical_Constitutive_Law_Base
    {
    public:
      Coupled_Magnetomechanical_Constitutive_Law_Base(
        const ConstitutiveParameters &constitutive_parameters);

      virtual void update_internal_data(const Tensor<1, dim> &         H,
                                        const SymmetricTensor<2, dim> &C,
                                        const DiscreteTime &time) = 0;

      // Free energy
      virtual double get_psi() const = 0;

      // Magnetic induction: B = - d_psi/d_H
      virtual Tensor<1, dim> get_B() const = 0;

      // Piola-Kirchhoff stress: S = 2 d_psi/d_C
      virtual SymmetricTensor<2, dim> get_S() const = 0;

      // Magnetostatic tangent: BB = dB/dH = - d2_psi/d_H.d_H
      virtual SymmetricTensor<2, dim> get_BB() const = 0;

      // Magnetoelastic coupling tangent: PP = -dS/dH = - 2 d2_psi/d_C.d_H
      virtual Tensor<3, dim> get_PP() const = 0;

      // Material elastic tangent: HH = 2 dS/dC = 4 d2_psi/d_C.d_C
      virtual SymmetricTensor<4, dim> get_HH() const = 0;


      virtual void update_end_of_timestep(){};

    protected:
      const ConstitutiveParameters &constitutive_parameters;

      // Shear modulus
      double get_mu_e() const;

      // Shear modulus at saturation magnetic field
      double get_mu_e_inf() const;

      // Saturation magnetic field strength
      double get_mu_e_h_sat() const;

      // Poisson ratio
      double get_nu_e() const;

      // Lam&eacute; parameter
      double get_lambda_e() const;

      // Bulk modulus
      double get_kappa_e() const;


      // Viscoelastic shear modulus
      double get_mu_v() const;

      // Viscoelastic shear modulus at magnetic saturation
      double get_mu_v_inf() const;

      // Saturation magnetic field strength for viscoelastic shear modulus
      double get_mu_v_h_sat() const;

      // Characteristic relaxation time
      double get_tau_v() const;

      // Relative magnetic permeability
      double get_mu_r() const;

      // Magnetic permeability constant
      constexpr double get_mu_0() const;
    };

    template <int dim>
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::
      Coupled_Magnetomechanical_Constitutive_Law_Base(
        const ConstitutiveParameters &constitutive_parameters)
      : constitutive_parameters(constitutive_parameters)
    {
      Assert(get_kappa_e() > 0, ExcInternalError());
    }



    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_e() const
    {
      return constitutive_parameters.mu_e;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_e_inf() const
    {
      return constitutive_parameters.mu_e_inf;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_e_h_sat() const
    {
      return constitutive_parameters.mu_e_h_sat;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_nu_e() const
    {
      return constitutive_parameters.nu_e;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_lambda_e() const
    {
      return 2.0 * get_mu_e() * get_nu_e() / (1.0 - 2.0 * get_nu_e());
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_kappa_e() const
    {
      return (2.0 * get_mu_e() * (1.0 + get_nu_e())) /
             (3.0 * (1.0 - 2.0 * get_nu_e()));
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_v() const
    {
      return constitutive_parameters.mu_v;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_v_inf() const
    {
      return constitutive_parameters.mu_v_inf;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_v_h_sat() const
    {
      return constitutive_parameters.mu_v_h_sat;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_tau_v() const
    {
      return constitutive_parameters.tau_v;
    }

    template <int dim>
    double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_r() const
    {
      return constitutive_parameters.mu_r;
    }

    template <int dim>
    constexpr double
    Coupled_Magnetomechanical_Constitutive_Law_Base<dim>::get_mu_0() const
    {
      return 4.0 * numbers::PI * 1e-7;
    }


    // @sect4{Magnetoelastic constitutive law (using automatic differentiation)}

    // @f[
    //   \psi_{0} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{1}{2} \mu_{e} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)
    //     \left[ tr(\mathbf{C}) - d - 2 \ln (det(\mathbf{F})) \right]
    // + \lambda_{e} \ln^{2} \left(det(\mathbf{F}) \right)
    // - \frac{1}{2} \mu_{0} \mu_{r} det(\mathbf{F})
    //     \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]
    // @f]
    // with
    // @f[
    //  f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)
    // = 1 + \left[ \frac{\mu_{e}^{\infty}}{\mu_{e}} - 1 \right]
    //     \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}}
    //       {\left(\mu_{e}^{sat}\right)^{2}} \right)
    // @f]
    // The variable $d = tr(\mathbf{I})$ represents the spatial dimension. 
    template <int dim, AD::NumberTypes ADTypeCode>
    class Magnetoelastic_Constitutive_Law_AD
      : public Coupled_Magnetomechanical_Constitutive_Law_Base<dim>
    {
      // Define the helper type that we will use in the AD computations for our
      // scalar energy function. Note that we expect it to return values of
      // type double.
      using ADHelper     = AD::ScalarFunction<dim, ADTypeCode, double>;
      using ADNumberType = typename ADHelper::ad_type;

    public:
      Magnetoelastic_Constitutive_Law_AD(
        const ConstitutiveParameters &constitutive_parameters);

      void update_internal_data(const Tensor<1, dim> &         H,
                                const SymmetricTensor<2, dim> &C,
                                const DiscreteTime &) override;

      // Free energy
      double get_psi() const override;

      // Magnetic induction: B = -dpsi/dH
      Tensor<1, dim> get_B() const override;

      // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
      SymmetricTensor<2, dim> get_S() const override;

      // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
      SymmetricTensor<2, dim> get_BB() const override;

      // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
      Tensor<3, dim> get_PP() const override;

      // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
      SymmetricTensor<4, dim> get_HH() const override;

    private:
      // Define some extractors that will help us set independent variables
      // and later get the computed values related to the dependent
      // variables. Each of these extractors is related to the gradient of a
      // component of the solution field (in this case, displacement and
      // magnetic scalar potential). Here "C" is the right Cauchy-Green
      // tensor and "H" is the magnetic field.
      const FEValuesExtractors::Vector             H_dofs;
      const FEValuesExtractors::SymmetricTensor<2> C_dofs;

      ADHelper           ad_helper;
      double             psi;
      Vector<double>     Dpsi;
      FullMatrix<double> D2psi;
    };


    template <int dim, AD::NumberTypes ADTypeCode>
    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::
      Magnetoelastic_Constitutive_Law_AD(
        const ConstitutiveParameters &constitutive_parameters)
      : Coupled_Magnetomechanical_Constitutive_Law_Base<dim>(
          constitutive_parameters)
      , H_dofs(0)
      , C_dofs(Tensor<1, dim>::n_independent_components)
      , ad_helper(Tensor<1, dim>::n_independent_components +
                  SymmetricTensor<2, dim>::
                    n_independent_components) // n_independent_variables
      , psi(0.0)
      , Dpsi(ad_helper.n_independent_variables())
      , D2psi(ad_helper.n_independent_variables(),
              ad_helper.n_independent_variables())
    {}


    template <int dim, AD::NumberTypes ADTypeCode>
    void
    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::update_internal_data(
      const Tensor<1, dim> &         H,
      const SymmetricTensor<2, dim> &C,
      const DiscreteTime &)
    {
      Assert(determinant(C) > 0, ExcInternalError());

      // Since we reuse this data structure at each time step,
      // we need to clear it of all stale information before
      // use.
      ad_helper.reset();

      // This is the "recording" phase of the operations.
      // First, we set the values for all fields.
      // These could happily be set to anything, unless the function will
      // be evaluated along a branch not otherwise traversed during later
      // use. For this reason, in this example instead of using some dummy
      // values, we'll actually map out the function at the same point
      // around which we'll later linearize it.
      ad_helper.register_independent_variable(H, H_dofs);
      ad_helper.register_independent_variable(C, C_dofs);
      // NOTE: We have to extract the sensitivities in the order we wish to
      // introduce them. So this means we have to do it by logical order
      // of the extractors that we've created.
      // TODO: Check if the note above is still true!
      const Tensor<1, dim, ADNumberType> H_AD =
        ad_helper.get_sensitive_variables(H_dofs);
      const SymmetricTensor<2, dim, ADNumberType> C_AD =
        ad_helper.get_sensitive_variables(C_dofs);
      const ADNumberType det_F_AD = std::sqrt(determinant(C_AD));
      const SymmetricTensor<2, dim, ADNumberType> C_inv_AD = invert(C_AD);

      // A scaling function that will cause the shear modulus
      // to change (increase) under the influence of a magnetic
      // field.
      const ADNumberType f_mu_e_AD =
        1.0 + (this->get_mu_e_inf() / this->get_mu_e() - 1.0) *
                std::tanh((2.0 * H_AD * H_AD) /
                          (this->get_mu_e_h_sat() * this->get_mu_e_h_sat()));

      // Here we define the material stored energy function.
      // This example is sufficiently complex to warrant the use of AD to,
      // at the very least, verify an unassisted implementation.
      const ADNumberType psi_AD =
        0.5 * this->get_mu_e() * f_mu_e_AD *
          (trace(C_AD) - dim - 2.0 * std::log(det_F_AD))                 //
        + this->get_lambda_e() * std::log(det_F_AD) * std::log(det_F_AD) //
        - 0.5 * this->get_mu_0() * this->get_mu_r() * det_F_AD *
            (H_AD * C_inv_AD * H_AD); //

      // Register the definition of the total stored energy
      ad_helper.register_dependent_variable(psi_AD);

      // Store the the gradient of the stored energy density function and
      // linearization. These are expensive to compute, so we'll do this once
      // and extract the desired values from these intermediate outputs.
      psi = ad_helper.compute_value();
      ad_helper.compute_gradient(Dpsi);
      ad_helper.compute_hessian(D2psi);
    }

    // Free energy
    template <int dim, AD::NumberTypes ADTypeCode>
    double Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::get_psi() const
    {
      return psi;
    }

    // Extract the desired components of the gradient vector and Hessian
    // matrix.
    // Magnetic induction: B = -dpsi/dH
    template <int dim, AD::NumberTypes ADTypeCode>
    Tensor<1, dim>

    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::get_B() const
    {
      const Tensor<1, dim> dpsi_dH =
        ad_helper.extract_gradient_component(Dpsi, H_dofs);
      return -dpsi_dH;
    }

    // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
    template <int dim, AD::NumberTypes ADTypeCode>
    SymmetricTensor<2, dim>

    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::get_S() const
    {
      const SymmetricTensor<2, dim> dpsi_dC =
        ad_helper.extract_gradient_component(Dpsi, C_dofs);
      return 2.0 * dpsi_dC;
    }

    // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
    template <int dim, AD::NumberTypes ADTypeCode>
    SymmetricTensor<2, dim>

    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::get_BB() const
    {
      const Tensor<2, dim> dpsi_dH_dH =
        ad_helper.extract_hessian_component(D2psi, H_dofs, H_dofs);
      return -symmetrize(dpsi_dH_dH);
    }

    // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
    // Here the order of the extractor
    // arguments is especially important, as it dictates the order in which
    // the directional derivatives are taken.
    template <int dim, AD::NumberTypes ADTypeCode>
    Tensor<3, dim>

    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::get_PP() const
    {
      const Tensor<3, dim> dpsi_dC_dH =
        ad_helper.extract_hessian_component(D2psi, C_dofs, H_dofs);
      return -2.0 * dpsi_dC_dH;
    }

    // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
    template <int dim, AD::NumberTypes ADTypeCode>
    SymmetricTensor<4, dim>

    Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode>::get_HH() const
    {
      const SymmetricTensor<4, dim> dpsi_dC_dC =
        ad_helper.extract_hessian_component(D2psi, C_dofs, C_dofs);
      return 4.0 * dpsi_dC_dC;
    }


    // @sect4{Magneto-viscoelastic constitutive law (using symbolic algebra and differentiation)}

    // Considering just a single dissipative mechanism `i`:
    // @f[
    //   \psi_{0} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = \psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // @f]
    // @f[
    //   \psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{1}{2} \mu_{e} f_{\mu_{e}^{ME}} \left( \boldsymbol{\mathbb{H}} \right)
    //     \left[ tr(\mathbf{C}) - d - 2 \ln (det(\mathbf{F})) \right]
    // + \lambda_{e} \ln^{2} \left(det(\mathbf{F}) \right)
    // - \frac{1}{2} \mu_{0} \mu_{r} det(\mathbf{F})
    //     \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]
    // @f]
    // @f[
    //   \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{1}{2} \mu_{v} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)
    //     \left[ \mathbf{C}_{v} : \left[
    //       \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //       \mathbf{C} \right] - d - \ln\left( det\left(\mathbf{C}_{v}\right)
    //       \right)  \right]
    // @f]
    // with
    // @f[
    //   f_{\mu_{e}}^{ME} \left( \boldsymbol{\mathbb{H}} \right)
    // = 1 + \left[ \frac{\mu_{e}^{\infty}}{\mu_{e}} - 1 \right]
    //     \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}}
    //       {\left(\mu_{e}^{sat}\right)^{2}} \right)
    // @f]
    // @f[
    //   f_{\mu_{v}}^{MVE} \left( \boldsymbol{\mathbb{H}} \right)
    // = 1 + \left[ \frac{\mu_{v}^{\infty}}{\mu_{v}} - 1 \right]
    //     \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}}
    //       {\left(\mu_{v}^{sat}\right)^{2}} \right)
    // @f]
    // and the evolution law
    // @f[
    //  \dot{\mathbf{C}_{v}}
    // = \frac{1}{\tau} \left[
    //       \left[\left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //         \mathbf{C}\right]^{-1}
    //     - \mathbf{C}_{v} \right]
    // @f]
    template <int dim>
    class Magnetoviscoelastic_Constitutive_Law_SD
      : public Coupled_Magnetomechanical_Constitutive_Law_Base<dim>
    {
    public:
      Magnetoviscoelastic_Constitutive_Law_SD(
        const ConstitutiveParameters &constitutive_parameters,
        const SD::OptimizerType       optimizer_type,
        const SD::OptimizationFlags   optimization_flags);

      void update_internal_data(const Tensor<1, dim> &         H,
                                const SymmetricTensor<2, dim> &C,
                                const DiscreteTime &           time) override;

      void update_end_of_timestep() override;

      // Free energy
      double get_psi() const override;

      // Magnetic induction: B = -dpsi/dH
      Tensor<1, dim> get_B() const override;

      // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
      SymmetricTensor<2, dim> get_S() const override;

      // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
      SymmetricTensor<2, dim> get_BB() const override;

      // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
      Tensor<3, dim> get_PP() const override;

      // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
      SymmetricTensor<4, dim> get_HH() const override;

    private:
      // Define some material parameters
      const SD::Expression mu_e_SD;
      const SD::Expression mu_e_inf_SD;
      const SD::Expression mu_e_h_sat_SD;
      const SD::Expression lambda_e_SD;
      const SD::Expression mu_v_SD;
      const SD::Expression mu_v_inf_SD;
      const SD::Expression mu_v_h_sat_SD;
      const SD::Expression tau_v_SD;
      const SD::Expression delta_t_SD;
      const SD::Expression mu_r_SD;

      // Define some independent variables
      const Tensor<1, dim, SD::Expression>          H_SD;
      const SymmetricTensor<2, dim, SD::Expression> C_SD;
      // Internal variables
      const SymmetricTensor<2, dim, SD::Expression> Q_t_SD;
      const SymmetricTensor<2, dim, SD::Expression> Q_t1_SD;

      // Dependent variables
      // Have to store these, as we require them to retreive
      // data from the optimizer.
      SD::Expression                          psi_SD;
      Tensor<1, dim, SD::Expression>          B_SD;
      SymmetricTensor<2, dim, SD::Expression> S_SD;
      SymmetricTensor<2, dim, SD::Expression> BB_SD;
      Tensor<3, dim, SD::Expression>          PP_SD;
      SymmetricTensor<4, dim, SD::Expression> HH_SD;

      // An optimizer to evaluate the dependent functions. As specified
      // by the template parameter, the numerical result will be of
      // type <tt>double</tt>.
      SD::BatchOptimizer<double> optimizer;

      // Store some numerical values.
      // Value of internal variable at this Newton step and timestep
      SymmetricTensor<2, dim> Q_t;
      // Value of internal variable at the previous timestep
      SymmetricTensor<2, dim> Q_t1;

      SD::types::substitution_map
      make_substitution_map(const Tensor<1, dim> &         H,
                            const SymmetricTensor<2, dim> &C,
                            const double                   delta_t) const;

      void initialize_optimizer();
    };


    template <int dim>
    Magnetoviscoelastic_Constitutive_Law_SD<dim>::
      Magnetoviscoelastic_Constitutive_Law_SD(
        const ConstitutiveParameters &constitutive_parameters,
        const SD::OptimizerType       optimizer_type,
        const SD::OptimizationFlags   optimization_flags)
      : Coupled_Magnetomechanical_Constitutive_Law_Base<dim>(
          constitutive_parameters)
      , mu_e_SD("mu_e")
      , mu_e_inf_SD("mu_e_inf")
      , mu_e_h_sat_SD("mu_e_h_sat")
      , lambda_e_SD("lambda_e")
      , mu_v_SD("mu_v")
      , mu_v_inf_SD("mu_v_inf")
      , mu_v_h_sat_SD("mu_v_h_sat")
      , tau_v_SD("tau_v")
      , delta_t_SD("delta_t")
      , mu_r_SD("mu_r")
      , H_SD(SD::make_vector_of_symbols<dim>("H"))
      , C_SD(SD::make_symmetric_tensor_of_symbols<2, dim>("C"))
      , Q_t_SD(SD::make_symmetric_tensor_of_symbols<2, dim>("Q_t"))
      , Q_t1_SD(SD::make_symmetric_tensor_of_symbols<2, dim>("Q_t1"))
      , optimizer(optimizer_type, optimization_flags)
      , Q_t(Physics::Elasticity::StandardTensors<dim>::I)
      , Q_t1(Physics::Elasticity::StandardTensors<dim>::I)
    {
      initialize_optimizer();
    }


    template <int dim>
    SD::types::substitution_map
    Magnetoviscoelastic_Constitutive_Law_SD<dim>::make_substitution_map(
      const Tensor<1, dim> &         H,
      const SymmetricTensor<2, dim> &C,
      const double                   delta_t) const
    {
      return SD::make_substitution_map(
        std::make_pair(mu_e_SD, this->get_mu_e()),
        std::make_pair(mu_e_inf_SD, this->get_mu_e_inf()),
        std::make_pair(mu_e_h_sat_SD, this->get_mu_e_h_sat()),
        std::make_pair(lambda_e_SD, this->get_lambda_e()),
        std::make_pair(mu_v_SD, this->get_mu_v()),
        std::make_pair(mu_v_inf_SD, this->get_mu_v_inf()),
        std::make_pair(mu_v_h_sat_SD, this->get_mu_v_h_sat()),
        std::make_pair(tau_v_SD, this->get_tau_v()),
        std::make_pair(delta_t_SD, delta_t),
        std::make_pair(mu_r_SD, this->get_mu_r()),
        std::make_pair(H_SD, H),
        std::make_pair(C_SD, C),
        std::make_pair(Q_t_SD, Q_t),
        std::make_pair(Q_t1_SD, Q_t1));
    }


    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law_SD<dim>::initialize_optimizer()
    {
      // These are expressions written in terms of C_SD,
      // a primary independent variable.
      const SD::Expression det_F_SD = std::sqrt(determinant(C_SD));
      const SymmetricTensor<2, dim, SD::Expression> C_inv_SD = invert(C_SD);

      // A scaling function that will cause the elastic shear modulus
      // to change (increase) under the influence of a magnetic
      // field.
      // @cite Pelteret2018a eq. 29
      const SD::Expression f_mu_e_SD =
        1.0 +
        (mu_e_inf_SD / mu_e_SD - 1.0) *
          std::tanh((2.0 * H_SD * H_SD) / (mu_e_h_sat_SD * mu_e_h_sat_SD));

      // Here we define the magneto-elastic contribution to the stored energy
      // function.
      const SD::Expression psi_ME_SD =
        0.5 * mu_e_SD * f_mu_e_SD *
          (trace(C_SD) - dim - 2.0 * std::log(det_F_SD)) +
        lambda_e_SD * std::log(det_F_SD) * std::log(det_F_SD) -
        0.5 * this->get_mu_0() * mu_r_SD * det_F_SD * (H_SD * C_inv_SD * H_SD);

      // Next we define the magneto-viscoelastic contribution to the stored
      // energy function. To the CAS, Q_t_SD appears to be independent of
      // C_SD, and so any derivatives wrt. C_SD will ignore this inherent
      // dependence. This means that deriving the function f = f(C,Q) wrt. C
      // will take partial derivatives.

      // A scaling function that will cause the viscous shear modulus
      // to change (increase) under the influence of a magnetic
      // field.
      // @cite Pelteret2018a eq. 29
      const SD::Expression f_mu_v_SD =
        1.0 +
        (mu_v_inf_SD / mu_v_SD - 1.0) *
          std::tanh((2.0 * H_SD * H_SD) / (mu_v_h_sat_SD * mu_v_h_sat_SD));

      // psi: See @cite Linder2011a eq. 46 / @cite Pelteret2018a eq. 28
      const SD::Expression psi_MVE_SD =
        0.5 * mu_v_SD * f_mu_v_SD *
        (Q_t_SD * (std::pow(det_F_SD, -2.0 / dim) * C_SD) - dim -
         std::log(determinant(Q_t_SD)));

      // Here we define the material's total free energy function.
      psi_SD = psi_ME_SD + psi_MVE_SD;

      // Compute some symbolic expressions that are dependent on the
      // independent variables. These could be, for example, scalar
      // expressions or tensors of expressions.
      B_SD = -SD::differentiate(psi_SD, H_SD);
      S_SD = 2.0 * SD::differentiate(psi_SD, C_SD);

      // Inform the CAS of the explicit dependency of Q_t_SD on C_SD,
      // i.e., state that Q_t_SD = Q_t_SD (C_SD).
      // This means that future differential operations wrt. C_SD will
      // take into account this dependence (i.e., compute total derivatives)
      // Since we now state that f = f(C,Q(C)).
      //
      // Evolution law: See @cite Linder2011a eq. 41
      // or @cite Pelteret2018a eq. 30
      // Discretising in time (BDF 1) gives us this expression,
      // i.e., @cite Linder2011a eq. 54
      const SymmetricTensor<2, dim, SD::Expression> Q_t_SD_explicit =
        (1.0 / (1.0 + delta_t_SD / tau_v_SD)) *
        (Q_t1_SD +
         (delta_t_SD / tau_v_SD * std::pow(det_F_SD, 2.0 / dim) * C_inv_SD));

      const SD::types::substitution_map substitution_map_explicit =
        SD::make_substitution_map(std::make_pair(Q_t_SD, Q_t_SD_explicit));

      BB_SD = symmetrize(
        SD::differentiate(substitute(B_SD, substitution_map_explicit), H_SD));
      PP_SD =
        -SD::differentiate(substitute(S_SD, substitution_map_explicit), H_SD);
      HH_SD =
        2.0 *
        SD::differentiate(substitute(S_SD, substitution_map_explicit), C_SD);

      // Now we need to tell the optimizer what entries we need to provide
      // numerical values for in order for it to successfully perform its
      // calculations. These are, collectively, the independent variables
      // for the problem, the history variables and the constitutive
      // parameters (since we've not hard encoded them in the energy
      // function).
      //
      // So what we really want is to provide it a collection of
      // symbols, which one could accomplish in this way:
      // @code
      // optimizer.register_symbols(SD::make_symbol_map(
      //   mu_e_SD, mu_e_inf_SD, mu_e_h_sat_SD, lambda_e_SD,
      //   mu_v_SD, mu_v_inf_SD, mu_v_h_sat_SD, tau_v_SD,
      //   delta_t_SD, mu_r_SD,
      //   H_SD, C_SD,
      //   Q_t_SD, Q_t1_SD));
      // @endcode
      // But this is all actually already encoded as the keys of the
      // substitution map. Doing the above would also mean that we
      // need to manage the symbols in two places (here and when constructing
      // the substitution map), which is annoying and a potential source of
      // error if this material class is modified or extended.
      // Since we're not interested in the values at this point,
      // it's OK if the substitution map is filled with invalid data
      // for the values associated with each key entry.
      optimizer.register_symbols(
        SD::Utilities::extract_symbols(make_substitution_map({}, {}, 0)));

      // We then inform the optimizer of what values we want calculated, which
      // in our situation encompasses all of the dependent variables (namely
      // the energy function and its various derivatives).
      optimizer.register_functions(psi_SD, B_SD, S_SD, BB_SD, PP_SD, HH_SD);

      // Now we determine an equivalent code path that will evaluate
      // all of the dependent functions at once, but with less computational
      // cost than when evaluating the symbolic expression directly.
      // Note: This is an expensive call, so we want execute it as few times
      // as possible. We've done it in the constructor of our class, which
      // achieves the goal of being called only once per class instance.
      optimizer.optimize();
    }


    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law_SD<dim>::update_internal_data(
      const Tensor<1, dim> &         H,
      const SymmetricTensor<2, dim> &C,
      const DiscreteTime &           time)
    {
      const double delta_t = time.get_previous_step_size();

      const double                  det_F = std::sqrt(determinant(C));
      const SymmetricTensor<2, dim> C_inv = invert(C);
      AssertThrow(det_F > 0.0, ExcInternalError());

      // Update internal history (Real values)
      // Evolution law: See @cite Linder2011a eq. 41
      // Discretising in time (BDF 1) gives us this expression,
      // i.e., @cite Linder2011a eq. 54
      Q_t = (1.0 / (1.0 + delta_t / this->get_tau_v())) *
            (Q_t1 + (delta_t / this->get_tau_v()) * std::pow(det_F, 2.0 / dim) *
                      C_inv);

      // Next we pass the optimizer the numeric values that we wish the
      // constitutive parameters and independent variables to represent.
      const auto substitution_map = make_substitution_map(H, C, delta_t);

      // When making this next call, the call path used to (numerically)
      // evaluate the dependent functions is quicker than dictionary
      // substitution.
      optimizer.substitute(substitution_map);
    }

    // Data extraction from the optimizer.
    // Note: When doing the evaluation, we need the exact expressions of
    // the data to extracted from the optimizer. The implication of this
    // is that we need to store the symbolic expressions of all dependent
    // variables for the lifetime of the optimizer (naturally, the same
    // is implied for the input variables). Furthermore, when serializing
    // a material class like this one we'd either need to serialize these
    // expressions as well or we'd need to reconstruct them upon reloading.

    // Free energy
    template <int dim>
    double Magnetoviscoelastic_Constitutive_Law_SD<dim>::get_psi() const
    {
      return optimizer.evaluate(psi_SD);
    }

    // Magnetic induction: B = -dpsi/dH
    template <int dim>
    Tensor<1, dim> Magnetoviscoelastic_Constitutive_Law_SD<dim>::get_B() const
    {
      return optimizer.evaluate(B_SD);
    }

    // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
    template <int dim>
    SymmetricTensor<2, dim>
    Magnetoviscoelastic_Constitutive_Law_SD<dim>::get_S() const
    {
      return optimizer.evaluate(S_SD);
    }

    // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
    template <int dim>
    SymmetricTensor<2, dim>
    Magnetoviscoelastic_Constitutive_Law_SD<dim>::get_BB() const
    {
      return optimizer.evaluate(BB_SD);
    }

    // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
    template <int dim>
    Tensor<3, dim> Magnetoviscoelastic_Constitutive_Law_SD<dim>::get_PP() const
    {
      return optimizer.evaluate(PP_SD);
    }

    // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
    template <int dim>
    SymmetricTensor<4, dim>
    Magnetoviscoelastic_Constitutive_Law_SD<dim>::get_HH() const
    {
      return optimizer.evaluate(HH_SD);
    }


    // Record value of history variable for use
    // as the "past value" at the next time step
    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law_SD<dim>::update_end_of_timestep()
    {
      Q_t1 = Q_t;
    };



    // @sect3{A more complex example (continued): Parameters and hand-derived material classes}
    //
    // We'll take the opportunity to present two different paradigms for
    // defining constitutive law classes. The second will provide more
    // flexibility than the first (thereby making it more easily extensible,
    // in the author's opinion) at the expense of some performance.

    // @sect4{Magnetoelastic constitutive law (hand-derived)}

    template <int dim>
    class Magnetoelastic_Constitutive_Law
      : public Coupled_Magnetomechanical_Constitutive_Law_Base<dim>
    {
    public:
      Magnetoelastic_Constitutive_Law(
        const ConstitutiveParameters &constitutive_parameters);

      void update_internal_data(const Tensor<1, dim> &         H,
                                const SymmetricTensor<2, dim> &C,
                                const DiscreteTime &) override;

      // Free energy
      double get_psi() const override;

      // Magnetic induction: B = -dpsi/dH
      Tensor<1, dim> get_B() const override;

      // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
      SymmetricTensor<2, dim> get_S() const override;

      // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
      SymmetricTensor<2, dim> get_BB() const override;

      // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
      Tensor<3, dim> get_PP() const override;

      // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
      SymmetricTensor<4, dim> get_HH() const override;

    private:
      double                  psi;
      Tensor<1, dim>          B;
      SymmetricTensor<2, dim> S;
      SymmetricTensor<2, dim> BB;
      Tensor<3, dim>          PP;
      SymmetricTensor<4, dim> HH;
    };


    template <int dim>
    Magnetoelastic_Constitutive_Law<dim>::Magnetoelastic_Constitutive_Law(
      const ConstitutiveParameters &constitutive_parameters)
      : Coupled_Magnetomechanical_Constitutive_Law_Base<dim>(
          constitutive_parameters)
      , psi(0.0)
    {}


    // From the free energy that, as mentioned earlier, is defined as
    // @f[
    //   \psi_{0} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{1}{2} \mu_{e} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)
    //     \left[ tr(\mathbf{C}) - d - 2 \ln (det(\mathbf{F})) \right]
    // + \lambda_{e} \ln^{2} \left(det(\mathbf{F}) \right)
    // - \frac{1}{2} \mu_{0} \mu_{r} det(\mathbf{F})
    //     \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]
    // @f]
    // with
    // @f[
    //  f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)
    // = 1 + \left[ \frac{\mu_{e}^{\infty}}{\mu_{e}} - 1 \right]
    //     \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}}
    //       {\left(\mu_{e}^{sat}\right)^{2}} \right) , \\
    // det(\mathbf{F}) = \sqrt{det(\mathbf{C})}
    // @f]
    // for this magneto-elastic material, the first derivatives that correspond
    // to the the magnetic induction vector and total Piola-Kirchhoff stress
    // tensor are
    // @f[
    //  \boldsymbol{\mathbb{B}} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // \dealcoloneq - \frac{d \psi_{0}}{d \boldsymbol{\mathbb{H}}}
    // = - \frac{1}{2} \mu_{e} \left[ tr(\mathbf{C}) - d - 2 \ln (det(\mathbf{F})) 
    //       \right] \frac{d f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}}}
    // + \mu_{0} \mu_{r} det(\mathbf{F}) \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}}
    //     \right]
    // @f]
    // @f{align}
    //  \mathbf{S}^{tot} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // \dealcoloneq 2 \frac{d \psi_{0} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)}{d \mathbf{C}}
    // &= \mu_{e} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right) 
    //     \left[ \frac{d\,tr(\mathbf{C})}{d \mathbf{C}} 
    //     - 2 \frac{1}{det(\mathbf{F})} \frac{d\,det(\mathbf{F})}{d \mathbf{C}} \right]
    // + 4 \lambda_{e} \ln \left(det(\mathbf{F}) \right) 
    //     \frac{1}{det(\mathbf{F})} \frac{d\,det(\mathbf{F})}{d \mathbf{C}} 
    // - \mu_{0} \mu_{r} \left[ 
    //     \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] 
    //     \frac{d\,det(\mathbf{F})}{d \mathbf{C}} + det(\mathbf{F}) 
    //     \frac{d \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} 
    //       \right]}{d \mathbf{C}} \right] \\
    // &= \mu_{e} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right) 
    //     \left[ \mathbf{I} - \mathbf{C}^{-1} \right]
    // + 2 \lambda_{e} \ln \left(det(\mathbf{F}) \right) \mathbf{C}^{-1}
    // - \mu_{0} \mu_{r} \left[ 
    //     \frac{1}{2}  \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] 
    //     det(\mathbf{F}) \mathbf{C}^{-1} 
    // - det(\mathbf{F}) 
    //     \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \otimes 
    //       \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \right]
    // @f}
    // with
    // @f[
    //   \frac{d f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}}}
    // = \left[ \frac{\mu_{e}^{\infty}}{\mu_{e}} - 1 \right] 
    //   \text{sech}^{2} \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}} 
    //     {\left(\mu_{e}^{sat}\right)^{2}} \right) 
    //   \left[ \frac{4} {\left(\mu_{e}^{sat}\right)^{2}} \boldsymbol{\mathbb{H}} \right]
    // @f]
    // @f[
    //   \frac{d\,tr(\mathbf{C})}{d \mathbf{C}} 
    // = \mathbf{I}
    // \quad \text{(the second-order identity tensor)}
    // @f]
    // @f[
    //   \frac{d\,det(\mathbf{F})}{d \mathbf{C}} 
    // = \frac{1}{2} det(\mathbf{F}) \mathbf{C}^{-1}
    // @f]
    // @f[
    // \frac{d C^{-1}_{ab}}{d C_{cd}}
    // = - sym\left( C^{-1}_{ac} C^{-1}_{bd} \right)
    // = -\frac{1}{2} \left[ C^{-1}_{ac} C^{-1}_{bd} + C^{-1}_{ad} C^{-1}_{bc} \right]
    // @f]
    // @f[
    //   \frac{d \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} 
    //   \right]}{d \mathbf{C}} 
    // = - \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \otimes 
    //   \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]
    // @f]
    // The use of the symmetry operator in the one derivation above helps to
    // ensure that the resulting rank-4 tensor, which holds minor symmetries
    // due to the symmetry of $\mathbf{C}$, still maps rank-2 symmetric
    // tensors to rank-2 symmetric tensors. See the SymmetricTensor class
    // documentation and the introduction to step-44 and for further explanation
    // as to what symmetry means in the context of fourth-order tensors.
    //
    // The linearization of each of the kinematic variables with respect to
    // their arguments are
    // @f[
    // \mathbb{D} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{d \boldsymbol{\mathbb{B}}}{d \boldsymbol{\mathbb{H}}} 
    // = - \frac{1}{2} \mu_{e} \left[ tr(\mathbf{C}) - d - 2 \ln (det(\mathbf{F})) 
    //     \right] \frac{d^{2} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}} \otimes d \boldsymbol{\mathbb{H}}}
    // + \mu_{0} \mu_{r} det(\mathbf{F}) \mathbf{C}^{-1}
    // @f]
    // @f{align}
    // \mathfrak{P}^{tot} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = - \frac{d \mathbf{S}^{tot}}{d \boldsymbol{\mathbb{H}}}
    // &= - \mu_{e} 
    //     \left[ \frac{d\,tr(\mathbf{C})}{d \mathbf{C}} 
    //     - 2 \frac{1}{det(\mathbf{F})} \frac{d\,det(\mathbf{F})}{d \mathbf{C}} \right] 
    //       \otimes \frac{d f_{\mu_{e} \left( \boldsymbol{\mathbb{H}} \right)}}{d \boldsymbol{\mathbb{H}}}
    // + \mu_{0} \mu_{r} \left[ 
    //     \frac{d\,det(\mathbf{F})}{d \mathbf{C}} \otimes
    //       \frac{d \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} 
    //         \right]}{d \boldsymbol{\mathbb{H}}} \right]
    // + det(\mathbf{F}) 
    //     \frac{d^{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} 
    //       \right]}{d \mathbf{C} \otimes \boldsymbol{\mathbb{H}}} \\
    // &= - \mu_{e} 
    //     \left[ \mathbf{I} - \mathbf{C}^{-1} \right] \otimes 
    //       \frac{d f_{\mu_{e} \left( \boldsymbol{\mathbb{H}} \right)}}{d \boldsymbol{\mathbb{H}}}
    // + \mu_{0} \mu_{r} \left[ 
    //     det(\mathbf{F}) \mathbf{C}^{-1} \otimes
    //       \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \right]
    // + det(\mathbf{F}) 
    //     \frac{d^{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} 
    //       \right]}{d \mathbf{C} \otimes \boldsymbol{\mathbb{H}}}
    // @f}
    // @f{align}
    // \mathcal{H}^{tot} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = 2 \frac{d \mathbf{S}^{tot}}{d \mathbf{C}}
    // &= 2 \mu_{e} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right) 
    //     \left[ - \frac{d \mathbf{C}^{-1}}{d \mathbf{C}} \right]
    //   + 4 \lambda_{e} \left[ \mathbf{C}^{-1} \otimes \left[ \frac{1}{det(\mathbf{F})} \frac{d \, det(\mathbf{F})}{d \mathbf{C}} \right] + \ln \left(det(\mathbf{F}) \right) \frac{d \mathbf{C}^{-1}}{d \mathbf{C}} \right] \\
    // &- \mu_{0} \mu_{r}  \left[ 
    //  det(\mathbf{F}) \mathbf{C}^{-1} \otimes \frac{d \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]}{d \mathbf{C}}
    // + \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \mathbf{C}^{-1} \otimes \frac{d \, det(\mathbf{F})}{d \mathbf{C}}
    // + \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] det(\mathbf{F}) \frac{d \mathbf{C}^{-1}}{d \mathbf{C}}
    // \right] \\
    // &+ 2 \mu_{0} \mu_{r} \left[ \left[
    //     \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \otimes 
    //       \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \right]
    //       \otimes \frac{d \, det(\mathbf{F})}{d \mathbf{C}}
    //     - det(\mathbf{F}) 
    //     \frac{d^{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}}\right]}{\mathbf{C} \otimes \mathbf{C}} 
    // \right] \\
    // &= 2 \mu_{e} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right) 
    //     \left[ - \frac{d \mathbf{C}^{-1}}{d \mathbf{C}} \right]
    //  + 4 \lambda_{e} \left[ \frac{1}{2} \mathbf{C}^{-1} \otimes \mathbf{C}^{-1} + \ln \left(det(\mathbf{F}) \right) \frac{d \mathbf{C}^{-1}}{d \mathbf{C}} \right] \\
    // &- \mu_{0} \mu_{r}  \left[ 
    //  - det(\mathbf{F}) \mathbf{C}^{-1} \otimes \left[ \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \otimes 
    //   \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \right]
    // + \frac{1}{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] det(\mathbf{F})  \mathbf{C}^{-1} \otimes \mathbf{C}^{-1}
    // + \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] det(\mathbf{F}) \frac{d \mathbf{C}^{-1}}{d \mathbf{C}}
    // \right] \\
    // &+ 2 \mu_{0} \mu_{r} \left[ \frac{1}{2} det(\mathbf{F}) \left[
    //     \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \otimes 
    //       \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \right]
    //       \otimes \mathbf{C}^{-1}
    //     - det(\mathbf{F}) 
    //     \frac{d^{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}}\right]}{\mathbf{C} \otimes \mathbf{C}} 
    // \right]
    // @f}
    // with
    // @f[
    //  \frac{d^{2} f_{\mu_{e}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}} \otimes d \boldsymbol{\mathbb{H}}}
    // = -2 \left[ \frac{\mu_{e}^{\infty}}{\mu_{e}} - 1 \right] 
    //   \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}} 
    //     {\left(\mu_{e}^{sat}\right)^{2}} \right) 
    //   \text{sech}^{2} \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}} 
    //     {\left(\mu_{e}^{sat}\right)^{2}} \right) 
    //   \left[ \frac{4} {\left(\mu_{e}^{sat}\right)^{2}} \mathbf{I} \right]
    // @f]
    // @f[
    // \frac{d \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} 
    //         \right]}{d \boldsymbol{\mathbb{H}}}
    // = 2 \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}}
    // @f]
    // @f[
    // \frac{d^{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}}\right]}{d \mathbf{C} \otimes d \boldsymbol{\mathbb{H}}}
    // \Rightarrow
    // \frac{d^{2} \left[ \mathbb{H}_{e} C^{-1}_{ef} \mathbb{H}_{f} 
    //       \right]}{d C_{ab} d \mathbb{H}_{c}}
    // = - C^{-1}_{ac} C^{-1}_{be} \mathbb{H}_{e} - C^{-1}_{ae} \mathbb{H}_{e} C^{-1}_{bc}
    // @f]
    // @f{align}
    // \frac{d^{2} \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}}\right]}{d \mathbf{C} \otimes d \mathbf{C}} 
    // &= -\frac{d \left[\left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right] \otimes 
    //       \left[ \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]\right]}{d \mathbf{C}} \\
    // \Rightarrow
    // \frac{d^{2} \left[ \mathbb{H}_{e} C^{-1}_{ef} \mathbb{H}_{f} 
    //       \right]}{d C_{ab} d C_{cd}}
    // &= sym\left( C^{-1}_{ae} \mathbb{H}_{e} C^{-1}_{cf} \mathbb{H}_{f} C^{-1}_{bd} 
    //           + C^{-1}_{ce} \mathbb{H}_{e} C^{-1}_{bf} \mathbb{H}_{f} C^{-1}_{ad} \right) \\
    // &= \frac{1}{2} \left[ 
    //      C^{-1}_{ae} \mathbb{H}_{e} C^{-1}_{cf} \mathbb{H}_{f} C^{-1}_{bd} 
    //    + C^{-1}_{ae} \mathbb{H}_{e} C^{-1}_{df} \mathbb{H}_{f} C^{-1}_{bc} 
    //    + C^{-1}_{ce} \mathbb{H}_{e} C^{-1}_{bf} \mathbb{H}_{f} C^{-1}_{ad}
    //    + C^{-1}_{be} \mathbb{H}_{e} C^{-1}_{df} \mathbb{H}_{f} C^{-1}_{ac}
    //   \right]
    // @f}
    //
    // In the method definition, we've composed these calculations slightly
    // differently. Some intermediate steps are also retained to give another perspective
    // of how to systematically compute the derivatives. Additionally, some
    // calculations are decomposed less or further to reuse some of the intermediate
    // values and, hopefully, aid the reader to follow the derivative operations.
    template <int dim>
    void Magnetoelastic_Constitutive_Law<dim>::update_internal_data(
      const Tensor<1, dim> &         H,
      const SymmetricTensor<2, dim> &C,
      const DiscreteTime &)
    {
      const double                  det_F = std::sqrt(determinant(C));
      const SymmetricTensor<2, dim> C_inv = invert(C);

      AssertThrow(det_F > 0.0, ExcInternalError());

      // A scaling function that will cause the shear modulus
      // to change (increase) under the influence of a magnetic
      // field.
      const double two_h_dot_h_div_h_sat_squ =
        (2.0 * H * H) / (this->get_mu_e_h_sat() * this->get_mu_e_h_sat());
      const double tanh_two_h_dot_h_div_h_sat_squ =
        std::tanh(two_h_dot_h_div_h_sat_squ);

      const double f_mu_e =
        1.0 + (this->get_mu_e_inf() / this->get_mu_e() - 1.0) *
                tanh_two_h_dot_h_div_h_sat_squ;

      // First derivative of scaling function
      const double dtanh_two_h_dot_h_div_h_sat_squ =
        std::pow(1.0 / std::cosh(two_h_dot_h_div_h_sat_squ),
                 2.0); // d/dx [tanh(x)] = sech^2(x)
      const Tensor<1, dim> dtwo_h_dot_h_div_h_sat_squ_dH =
        2.0 * 2.0 / (this->get_mu_e_h_sat() * this->get_mu_e_h_sat()) * H;

      const Tensor<1, dim> df_mu_e_dH =
        (this->get_mu_e_inf() / this->get_mu_e() - 1.0) *
        (dtanh_two_h_dot_h_div_h_sat_squ * dtwo_h_dot_h_div_h_sat_squ_dH);

      // Second derivative of scaling function
      const double d2tanh_two_h_dot_h_div_h_sat_squ =
        -2.0 * tanh_two_h_dot_h_div_h_sat_squ *
        dtanh_two_h_dot_h_div_h_sat_squ; // d/dx [sech^2(x)] = -2 * tanh(x) *
                                         // sech^2(x)
      const SymmetricTensor<2, dim> d2two_h_dot_h_div_h_sat_squ_dH_dH =
        2.0 * 2.0 / (this->get_mu_e_h_sat() * this->get_mu_e_h_sat()) *
        Physics::Elasticity::StandardTensors<dim>::I;

      const SymmetricTensor<2, dim> d2f_mu_e_dH_dH =
        (this->get_mu_e_inf() / this->get_mu_e() - 1.0) *
        (d2tanh_two_h_dot_h_div_h_sat_squ *
           symmetrize(outer_product(dtwo_h_dot_h_div_h_sat_squ_dH,
                                    dtwo_h_dot_h_div_h_sat_squ_dH)) +
         dtanh_two_h_dot_h_div_h_sat_squ * d2two_h_dot_h_div_h_sat_squ_dH_dH);

      // Some intermediate kinematic quantities
      const double         log_det_F         = std::log(det_F);
      const double         tr_C              = trace(C);
      const Tensor<1, dim> C_inv_dot_H       = C_inv * H;
      const double         H_dot_C_inv_dot_H = H * C_inv_dot_H;

      // First derivatives of kinematic quantities
      const SymmetricTensor<2, dim> d_tr_C_dC =
        Physics::Elasticity::StandardTensors<dim>::I;
      const SymmetricTensor<2, dim> ddet_F_dC     = 0.5 * det_F * C_inv;
      const SymmetricTensor<2, dim> dlog_det_F_dC = 0.5 * C_inv;

      const Tensor<1, dim> dH_dot_C_inv_dot_H_dH = 2.0 * C_inv_dot_H;

      SymmetricTensor<4, dim> dC_inv_dC;
      for (unsigned int A = 0; A < dim; ++A)
        for (unsigned int B = A; B < dim; ++B)
          for (unsigned int C = 0; C < dim; ++C)
            for (unsigned int D = C; D < dim; ++D)
              dC_inv_dC[A][B][C][D] -=               //
                0.5 * (C_inv[A][C] * C_inv[B][D]     //
                       + C_inv[A][D] * C_inv[B][C]); //

      const SymmetricTensor<2, dim> dH_dot_C_inv_dot_H_dC =
        -symmetrize(outer_product(C_inv_dot_H, C_inv_dot_H));

      // Second derivatives of kinematic quantities
      const SymmetricTensor<4, dim> d2log_det_F_dC_dC = 0.5 * dC_inv_dC;

      const SymmetricTensor<4, dim> d2det_F_dC_dC =
        0.5 * (outer_product(C_inv, ddet_F_dC) + det_F * dC_inv_dC);

      const SymmetricTensor<2, dim> d2H_dot_C_inv_dot_H_dH_dH = 2.0 * C_inv;

      Tensor<3, dim> d2H_dot_C_inv_dot_H_dC_dH;
      for (unsigned int A = 0; A < dim; ++A)
        for (unsigned int B = 0; B < dim; ++B)
          for (unsigned int C = 0; C < dim; ++C)
            d2H_dot_C_inv_dot_H_dC_dH[A][B][C] -=
              C_inv[A][C] * C_inv_dot_H[B] + //
              C_inv_dot_H[A] * C_inv[B][C];  //

      SymmetricTensor<4, dim> d2H_dot_C_inv_dot_H_dC_dC;
      for (unsigned int A = 0; A < dim; ++A)
        for (unsigned int B = A; B < dim; ++B)
          for (unsigned int C = 0; C < dim; ++C)
            for (unsigned int D = C; D < dim; ++D)
              d2H_dot_C_inv_dot_H_dC_dC[A][B][C][D] +=
                0.5 * (C_inv_dot_H[A] * C_inv_dot_H[C] * C_inv[B][D] +
                       C_inv_dot_H[A] * C_inv_dot_H[D] * C_inv[B][C] +
                       C_inv_dot_H[B] * C_inv_dot_H[C] * C_inv[A][D] +
                       C_inv_dot_H[B] * C_inv_dot_H[D] * C_inv[A][C]);

      // Free energy function
      psi =
        (0.5 * this->get_mu_e() * f_mu_e) *
          (tr_C - dim - 2.0 * std::log(det_F)) +
        this->get_lambda_e() * (std::log(det_F) * std::log(det_F)) -
        (0.5 * this->get_mu_0() * this->get_mu_r()) * det_F * (H * C_inv * H);

      // Magnetic induction
      B = -(0.5 * this->get_mu_e() * (tr_C - dim - 2.0 * log_det_F)) *
            df_mu_e_dH //
          + 0.5 * this->get_mu_0() * this->get_mu_r() * det_F *
              dH_dot_C_inv_dot_H_dH; //

      // Magnetostatic tangent
      BB = -(0.5 * this->get_mu_e() * (tr_C - dim - 2.0 * log_det_F)) * //
             d2f_mu_e_dH_dH                                             //
           + 0.5 * this->get_mu_0() * this->get_mu_r() * det_F *
               d2H_dot_C_inv_dot_H_dH_dH; //

      // Piola-Kirchhoff stress
      S = 2.0 * (0.5 * this->get_mu_e() * f_mu_e) *                        //
            (d_tr_C_dC - 2.0 * dlog_det_F_dC)                              //
          + 2.0 * this->get_lambda_e() * (2.0 * log_det_F * dlog_det_F_dC) //
          - 2.0 * (0.5 * this->get_mu_0() * this->get_mu_r()) *            //
              (H_dot_C_inv_dot_H * ddet_F_dC                               //
               + det_F * dH_dot_C_inv_dot_H_dC);                           //

      // Magnetoelastic coupling tangent: PP = -dS/dH
      PP = -2.0 * (0.5 * this->get_mu_e()) *                                  //
             outer_product(Tensor<2, dim>(d_tr_C_dC - 2.0 * dlog_det_F_dC),   //
                           df_mu_e_dH)                                        //
           +                                                                  //
           2.0 * (0.5 * this->get_mu_0() * this->get_mu_r()) *                //
             (outer_product(Tensor<2, dim>(ddet_F_dC), dH_dot_C_inv_dot_H_dH) //
              + det_F * d2H_dot_C_inv_dot_H_dC_dH);                           //

      // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
      HH =
        4.0 * (0.5 * this->get_mu_e() * f_mu_e) * (-2.0 * d2log_det_F_dC_dC) //
        + 4.0 * this->get_lambda_e() *                                       //
            (2.0 * outer_product(dlog_det_F_dC, dlog_det_F_dC)               //
             + 2.0 * log_det_F * d2log_det_F_dC_dC)                          //
        - 4.0 * (0.5 * this->get_mu_0() * this->get_mu_r()) *                //
            (H_dot_C_inv_dot_H * d2det_F_dC_dC                               //
             + outer_product(ddet_F_dC, dH_dot_C_inv_dot_H_dC)               //
             + outer_product(dH_dot_C_inv_dot_H_dC, ddet_F_dC)               //
             + det_F * d2H_dot_C_inv_dot_H_dC_dC);                           //
    }

    // Free energy
    template <int dim>
    double Magnetoelastic_Constitutive_Law<dim>::get_psi() const
    {
      return psi;
    }

    // Magnetic induction: B = -dpsi/dH
    template <int dim>
    Tensor<1, dim> Magnetoelastic_Constitutive_Law<dim>::get_B() const
    {
      return B;
    }

    // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
    template <int dim>
    SymmetricTensor<2, dim> Magnetoelastic_Constitutive_Law<dim>::get_S() const
    {
      return S;
    }

    // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
    template <int dim>
    SymmetricTensor<2, dim> Magnetoelastic_Constitutive_Law<dim>::get_BB() const
    {
      return BB;
    }

    // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
    template <int dim>
    Tensor<3, dim> Magnetoelastic_Constitutive_Law<dim>::get_PP() const
    {
      return PP;
    }

    // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
    template <int dim>
    SymmetricTensor<4, dim> Magnetoelastic_Constitutive_Law<dim>::get_HH() const
    {
      return HH;
    }


    // @sect4{Magneto-viscoelastic constitutive law (hand-derived)}

    template <int dim>
    class Magnetoviscoelastic_Constitutive_Law
      : public Coupled_Magnetomechanical_Constitutive_Law_Base<dim>
    {
    public:
      Magnetoviscoelastic_Constitutive_Law(
        const ConstitutiveParameters &constitutive_parameters);

      void update_internal_data(const Tensor<1, dim> &         H,
                                const SymmetricTensor<2, dim> &C,
                                const DiscreteTime &           time) override;

      // Free energy
      double get_psi() const override;

      // Magnetic induction: B = -dpsi/dH
      Tensor<1, dim> get_B() const override;

      // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
      SymmetricTensor<2, dim> get_S() const override;

      // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
      SymmetricTensor<2, dim> get_BB() const override;

      // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
      Tensor<3, dim> get_PP() const override;

      // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
      SymmetricTensor<4, dim> get_HH() const override;

      void update_end_of_timestep() override;

    private:
      double                  psi;
      Tensor<1, dim>          B;
      SymmetricTensor<2, dim> S;
      SymmetricTensor<2, dim> BB;
      Tensor<3, dim>          PP;
      SymmetricTensor<4, dim> HH;

      SymmetricTensor<2, dim>
        Q_t; // Value of internal variable at this Newton step and timestep
      SymmetricTensor<2, dim>
        Q_t1; // Value of internal variable at the previous timestep

      mutable GeneralDataStorage cache;

      void set_primary_variables(const Tensor<1, dim> &         H,
                                 const SymmetricTensor<2, dim> &C) const;

      void update_internal_variable(const DiscreteTime &time);

      // =========
      // Primary variables

      const Tensor<1, dim> &get_H() const;

      const SymmetricTensor<2, dim> &get_C() const;

      // =========
      // Scaling function and its derivatives

      double get_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const;

      double get_tanh_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const;

      // A scaling function that will cause the shear modulus
      // to change (increase) under the influence of a magnetic
      // field.
      double get_f_mu(const double mu,
                      const double mu_inf,
                      const double mu_h_sat) const;

      // First derivative of scaling function
      double get_dtanh_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const;

      Tensor<1, dim>
      get_dtwo_h_dot_h_div_h_sat_squ_dH(const double mu_h_sat) const;

      Tensor<1, dim> get_df_mu_dH(const double mu,
                                  const double mu_inf,
                                  const double mu_h_sat) const;

      // Second derivative of scaling function
      double get_d2tanh_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const;

      SymmetricTensor<2, dim>
      get_d2two_h_dot_h_div_h_sat_squ_dH_dH(const double mu_h_sat) const;

      SymmetricTensor<2, dim> get_d2f_mu_dH_dH(const double mu,
                                               const double mu_inf,
                                               const double mu_h_sat) const;

      // =========
      // Intermediate values directly attained from primary variables

      const double &get_det_F() const;

      const SymmetricTensor<2, dim> &get_C_inv() const;

      // =========

      const double &get_log_det_F() const;

      const double &get_trace_C() const;

      const Tensor<1, dim> &get_C_inv_dot_H() const;

      const double &get_H_dot_C_inv_dot_H() const;

      // =========
      // First derivatives

      // Derivative of internal variable wrt. field variables
      const SymmetricTensor<4, dim> &
      get_dQ_t_dC(const DiscreteTime &time) const;

      const SymmetricTensor<4, dim> &get_dC_inv_dC() const;

      const SymmetricTensor<2, dim> &get_d_tr_C_dC() const;

      const SymmetricTensor<2, dim> &get_ddet_F_dC() const;

      const SymmetricTensor<2, dim> &get_dlog_det_F_dC() const;

      const Tensor<1, dim> &get_dH_dot_C_inv_dot_H_dH() const;


      const SymmetricTensor<2, dim> &get_dH_dot_C_inv_dot_H_dC() const;

      // =========
      // Second derivatives

      const SymmetricTensor<4, dim> &get_d2log_det_F_dC_dC() const;

      const SymmetricTensor<4, dim> &get_d2det_F_dC_dC() const;

      const SymmetricTensor<2, dim> &get_d2H_dot_C_inv_dot_H_dH_dH() const;

      const Tensor<3, dim> &get_d2H_dot_C_inv_dot_H_dC_dH() const;

      const SymmetricTensor<4, dim> &get_d2H_dot_C_inv_dot_H_dC_dC() const;
    };


    template <int dim>
    Magnetoviscoelastic_Constitutive_Law<
      dim>::Magnetoviscoelastic_Constitutive_Law(const ConstitutiveParameters
                                                   &constitutive_parameters)
      : Coupled_Magnetomechanical_Constitutive_Law_Base<dim>(
          constitutive_parameters)
      , psi(0.0)
      , Q_t(Physics::Elasticity::StandardTensors<dim>::I)
      , Q_t1(Physics::Elasticity::StandardTensors<dim>::I)
    {}


    // As mentioned before, the free energy for the magneto-viscoelastic material
    // with one dissipative mechanism that we'll be considering is defined as
    // @f[
    //   \psi_{0} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = \psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // @f]
    // @f[
    //   \psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{1}{2} \mu_{e} f_{\mu_{e}^{ME}} \left( \boldsymbol{\mathbb{H}} \right)
    //     \left[ tr(\mathbf{C}) - d - 2 \ln (det(\mathbf{F})) \right]
    // + \lambda_{e} \ln^{2} \left(det(\mathbf{F}) \right)
    // - \frac{1}{2} \mu_{0} \mu_{r} det(\mathbf{F})
    //     \left[ \boldsymbol{\mathbb{H}} \cdot \mathbf{C}^{-1} \cdot \boldsymbol{\mathbb{H}} \right]
    // @f]
    // @f[
    //   \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{1}{2} \mu_{v} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)
    //     \left[ \mathbf{C}_{v} : \left[
    //       \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //       \mathbf{C} \right] - d - \ln\left( det\left(\mathbf{C}_{v}\right)
    //       \right)  \right]
    // @f]
    // with
    // @f[
    //   f_{\mu_{e}}^{ME} \left( \boldsymbol{\mathbb{H}} \right)
    // = 1 + \left[ \frac{\mu_{e}^{\infty}}{\mu_{e}} - 1 \right]
    //     \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}}
    //       {\left(\mu_{e}^{sat}\right)^{2}} \right)
    // @f]
    // @f[
    //   f_{\mu_{v}}^{MVE} \left( \boldsymbol{\mathbb{H}} \right)
    // = 1 + \left[ \frac{\mu_{v}^{\infty}}{\mu_{v}} - 1 \right]
    //     \tanh \left( 2 \frac{\boldsymbol{\mathbb{H}} \cdot \boldsymbol{\mathbb{H}}}
    //       {\left(\mu_{v}^{sat}\right)^{2}} \right)
    // @f]
    // and the evolution law
    // @f[
    //  \dot{\mathbf{C}_{v}}
    // = \frac{1}{\tau} \left[
    //       \left[\left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //         \mathbf{C}\right]^{-1}
    //     - \mathbf{C}_{v} \right]
    // @f]
    // By design, the magneto-elastic part of the energy 
    // $\psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)$
    // is identical to that of the magneto-elastic material presented earlier.
    // So, for the derivatives of the various contributions stemming from this
    // part of the energy, please refer to the previous section. We'll continue
    // to highlight the specific contributions from those terms by superscripting
    // the salient terms with $ME$. Furthermore, the magnetic saturation function
    // $f_{\mu_{v}}^{MVE} \left( \boldsymbol{\mathbb{H}} \right)$ for the damping term has
    // the identical form as that of the elastic term 
    // (i.e., $f_{\mu_{e}}^{ME} \left( \boldsymbol{\mathbb{H}} \right)$), and so the structure
    // of its derivatives are identical to that seen before; the only change is
    // for the three constitutive parameters that are now associated with
    // the viscous shear modulus $\mu_{v}$ rather than the elastic shear
    // modulus $\mu_{e}$.
    //
    // For this magneto-viscoelastic material, the first derivatives that correspond
    // to the the magnetic induction vector and total Piola-Kirchhoff stress
    // tensor are
    // @f[
    //  \boldsymbol{\mathbb{B}} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // \dealcoloneq - \frac{\partial \psi_{0} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}}} \Big\vert_{\mathbf{C}, \mathbf{C}_{v}}
    // \equiv \boldsymbol{\mathbb{B}}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \boldsymbol{\mathbb{B}}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // =  - \frac{d \psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}}} 
    //    - \frac{\partial \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}}}
    // @f]
    // @f[
    //  \mathbf{S}^{tot} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // \dealcoloneq 2 \frac{\partial \psi_{0} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)}{\partial \mathbf{C}} \Big\vert_{\mathbf{C}_{v}, \boldsymbol{\mathbb{H}}}
    // \equiv \mathbf{S}^{tot, ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \mathbf{S}^{tot, MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}}
    //     \right)
    // =  2 \frac{d \psi_{0}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)}{d \mathbf{C}} 
    //  + 2 \frac{\partial \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)}{\partial \mathbf{C}}
    // @f]
    // with the viscous contributions being
    // @f[
    //   \boldsymbol{\mathbb{B}}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = - \frac{\partial \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}}} \Big\vert_{\mathbf{C}, \mathbf{C}_{v}}
    // = - \frac{1}{2} \mu_{v}
    //     \left[ \mathbf{C}_{v} : \left[
    //       \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //       \mathbf{C} \right] - d - \ln\left( det\left(\mathbf{C}_{v}\right)
    //       \right)  \right] \frac{\partial f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}}}
    // @f]
    // @f[
    //   \mathbf{S}^{tot, MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}}
    //     \right)
    // = 2 \frac{\partial \psi_{0}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)}{\partial \mathbf{C}} \Big\vert_{\mathbf{C}_{v}, \boldsymbol{\mathbb{H}}}
    // = \mu_{v} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)
    //        \left[  \left[ \mathbf{C}_{v} : \mathbf{C} \right] \left[ - \frac{1}{d} \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}^{-1} \right]
    //        + \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}_{v} 
    //  \right]
    // @f]
    // and with
    // @f[ 
    // \frac{\partial f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}}}
    // \equiv \frac{d f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}}}
    // @f]
    //
    // At this point, we need to consider the time discretization of the evolution
    // law for the internal viscous variable, as it will dictate what the linearization
    // of the internal variable with respect to the field variables looks like.
    // Choosing the implicit first-order backwards difference scheme, then
    // @f[
    //  \dot{\mathbf{C}_{v}}
    // \approx \frac{\mathbf{C}_{v}^{(t)} - \mathbf{C}_{v}^{(t-1)}}{\Delta t}
    // = \frac{1}{\tau} \left[
    //       \left[\left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //         \mathbf{C}\right]^{-1}
    //     - \mathbf{C}_{v}^{(t)} \right]
    // @f]
    // where the superscript $(t)$ denotes that the quantity is taken at the current
    // timestep, and $(t-1)$ denotes quantities taken at the previous timestep
    // (i.e. a history variable). The timestep size $\Delta t$ is the difference
    // between the current time and that of the previous timestep.
    // Rearranging the terms so that all internal variable quantities at the
    // current time are on the left hand side of the equation, we get
    // @f[
    // \mathbf{C}_{v}^{(t)}
    // = \frac{1}{1 + \frac{\Delta t}{\tau_{v}}} \left[ 
    //     \mathbf{C}_{v}^{(t-1)} 
    //   + \frac{\Delta t}{\tau_{v}} 
    //     \left[\left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C} \right]^{-1} 
    //   \right]
    // @f]
    // that matches @cite Linder2011a equation 54.
    //
    // The linearization of each with respect to their arguments are
    // @f[
    // \mathbb{D} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{d \boldsymbol{\mathbb{B}}}{d \boldsymbol{\mathbb{H}}} 
    // \equiv \mathbb{D}^{ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \mathbb{D}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = \frac{d \boldsymbol{\mathbb{B}}^{ME}}{d \boldsymbol{\mathbb{H}}} 
    // + \frac{d \boldsymbol{\mathbb{B}}^{MVE}}{d \boldsymbol{\mathbb{H}}}
    // @f]
    // @f[
    // \mathfrak{P}^{tot} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = - \frac{d \mathbf{S}^{tot}}{d \boldsymbol{\mathbb{H}}}
    // \equiv \mathfrak{P}^{tot, ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \mathfrak{P}^{tot, MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = - \frac{d \mathbf{S}^{tot, ME}}{d \boldsymbol{\mathbb{H}}}
    // - \frac{d \mathbf{S}^{tot, MVE}}{d \boldsymbol{\mathbb{H}}}
    // @f]
    // @f[
    // \mathcal{H}^{tot} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = 2 \frac{d \mathbf{S}^{tot}}{d \mathbf{C}}
    // \equiv \mathcal{H}^{tot, ME} \left( \mathbf{C}, \boldsymbol{\mathbb{H}} \right)
    // + \mathcal{H}^{tot, MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = 2 \frac{d \mathbf{S}^{tot, ME}}{d \mathbf{C}}
    // + 2 \frac{d \mathbf{S}^{tot, MVE}}{d \mathbf{C}}
    // @f]
    // where the tangents for the viscous contributions are
    // @f[
    // \mathbb{D}^{MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = - \frac{1}{2} \mu_{v}
    //     \left[ \mathbf{C}_{v} : \left[
    //       \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}}
    //       \mathbf{C} \right] - d - \ln\left( det\left(\mathbf{C}_{v}\right)
    //       \right)  \right] \frac{\partial^{2} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}} \otimes \partial \boldsymbol{\mathbb{H}}}
    // @f]
    // @f[
    // \mathfrak{P}^{tot, MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // = - \mu_{v}
    //        \left[  \left[ \mathbf{C}_{v} : \mathbf{C} \right] \left[ - \frac{1}{d} \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}^{-1} \right]
    //        + \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}_{v} 
    //  \right] \otimes \frac{d f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}}}
    // @f]
    // @f{align}
    // \mathcal{H}^{tot, MVE} \left( \mathbf{C}, \mathbf{C}_{v}, \boldsymbol{\mathbb{H}} \right)
    // &= 2 \mu_{v} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)
    //   \left[ - \frac{1}{d} \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}^{-1} \right] \otimes
    //   \left[ \mathbf{C}_{v} + \mathbf{C} : \frac{d \mathbf{C}_{v}}{d \mathbf{C}} \right] \\
    // &+ 2 \mu_{v} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right) \left[ \mathbf{C}_{v} : \mathbf{C} \right]
    //   \left[ 
    //     \frac{1}{d^{2}} \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}^{-1} \otimes \mathbf{C}^{-1}
    //     - \frac{1}{d} \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \frac{d \mathbf{C}^{-1}}{d \mathbf{C}}
    //   \right] \\
    // &+ 2 \mu_{v} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)
    //   \left[ 
    //     -\frac{1}{d} \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \mathbf{C}_{v} \otimes \mathbf{C}^{-1}
    //     + \left[det\left(\mathbf{F}\right)\right]^{-\frac{2}{d}} \frac{d \mathbf{C}_{v}}{d \mathbf{C}} 
    //   \right]
    // @f}
    // with
    // @f[
    // \frac{\partial^{2} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{\partial \boldsymbol{\mathbb{H}} \otimes \partial \boldsymbol{\mathbb{H}}}
    // \equiv \frac{d^{2} f_{\mu_{v}^{MVE}} \left( \boldsymbol{\mathbb{H}} \right)}{d \boldsymbol{\mathbb{H}} \otimes d \boldsymbol{\mathbb{H}}}
    // @f]
    // and, from the evolution law,
    // @f[ 
    // \frac{d \mathbf{C}_{v}}{d \mathbf{C}} 
    // \equiv \frac{d \mathbf{C}_{v}^{(t)}}{d \mathbf{C}}
    //  = \frac{\frac{\Delta t}{\tau_{v}} }{1 + \frac{\Delta t}{\tau_{v}}} \left[ 
    //     \frac{1}{d} \left[det\left(\mathbf{F}\right)\right]^{\frac{2}{d}} \mathbf{C}^{-1} \otimes \mathbf{C}^{-1}
    //    + \left[det\left(\mathbf{F}\right)\right]^{\frac{2}{d}} \frac{d \mathbf{C}^{-1}}{d \mathbf{C}}
    //   \right]
    // @f]
    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law<dim>::update_internal_data(
      const Tensor<1, dim> &         H,
      const SymmetricTensor<2, dim> &C,
      const DiscreteTime &           time)
    {
      set_primary_variables(H, C);

      // Update internal variable based on new strain state
      update_internal_variable(time);

      // A scaling function that will cause the shear modulus
      // to change (increase) under the influence of a magnetic
      // field.
      const double f_mu_e = get_f_mu(this->get_mu_e(),
                                     this->get_mu_e_inf(),
                                     this->get_mu_e_h_sat());

      const double f_mu_v = get_f_mu(this->get_mu_v(),
                                     this->get_mu_v_inf(),
                                     this->get_mu_v_h_sat());

      // Scaling function first derivative
      const Tensor<1, dim> df_mu_e_dH = get_df_mu_dH(this->get_mu_e(),
                                                     this->get_mu_e_inf(),
                                                     this->get_mu_e_h_sat());

      const Tensor<1, dim> df_mu_v_dH = get_df_mu_dH(this->get_mu_v(),
                                                     this->get_mu_v_inf(),
                                                     this->get_mu_v_h_sat());


      // Scaling function second derivative
      const SymmetricTensor<2, dim> d2f_mu_e_dH_dH =
        get_d2f_mu_dH_dH(this->get_mu_e(),
                         this->get_mu_e_inf(),
                         this->get_mu_e_h_sat());

      const SymmetricTensor<2, dim> d2f_mu_v_dH_dH =
        get_d2f_mu_dH_dH(this->get_mu_v(),
                         this->get_mu_v_inf(),
                         this->get_mu_v_h_sat());

      // Some intermediate kinematic quantities
      const double &                 det_F = get_det_F();
      const SymmetricTensor<2, dim> &C_inv = get_C_inv();

      const double &log_det_F         = get_log_det_F();
      const double &tr_C              = get_trace_C();
      const double &H_dot_C_inv_dot_H = get_H_dot_C_inv_dot_H();

      // First derivatives of kinematic quantities
      const SymmetricTensor<2, dim> &d_tr_C_dC     = get_d_tr_C_dC();
      const SymmetricTensor<2, dim> &ddet_F_dC     = get_ddet_F_dC();
      const SymmetricTensor<2, dim> &dlog_det_F_dC = get_dlog_det_F_dC();

      // Derivative of internal variable wrt. field variables
      const SymmetricTensor<4, dim> &dQ_t_dC = get_dQ_t_dC(time);

      const Tensor<1, dim> &dH_dot_C_inv_dot_H_dH = get_dH_dot_C_inv_dot_H_dH();

      const SymmetricTensor<2, dim> &dH_dot_C_inv_dot_H_dC =
        get_dH_dot_C_inv_dot_H_dC();

      // Second derivatives of kinematic quantities
      const SymmetricTensor<4, dim> &d2log_det_F_dC_dC =
        get_d2log_det_F_dC_dC();

      const SymmetricTensor<4, dim> &d2det_F_dC_dC = get_d2det_F_dC_dC();

      const SymmetricTensor<2, dim> &d2H_dot_C_inv_dot_H_dH_dH =
        get_d2H_dot_C_inv_dot_H_dH_dH();

      const Tensor<3, dim> &d2H_dot_C_inv_dot_H_dC_dH =
        get_d2H_dot_C_inv_dot_H_dC_dH();

      const SymmetricTensor<4, dim> &d2H_dot_C_inv_dot_H_dC_dC =
        get_d2H_dot_C_inv_dot_H_dC_dC();


      // Free energy function
      psi = (0.5 * this->get_mu_e() * f_mu_e) *
              (tr_C - dim - 2.0 * std::log(det_F)) +
            this->get_lambda_e() * (std::log(det_F) * std::log(det_F));
      psi += (0.5 * this->get_mu_v() * f_mu_v) *
             (Q_t * (std::pow(det_F, -2.0 / dim) * C) - dim -
              std::log(determinant(Q_t)));
      psi -=
        (0.5 * this->get_mu_0() * this->get_mu_r()) * det_F * (H * C_inv * H);

      // Magnetic induction
      B =
        -(0.5 * this->get_mu_e() * (tr_C - dim - 2.0 * log_det_F)) * df_mu_e_dH;
      B -= (0.5 * this->get_mu_v()) *
           (Q_t * (std::pow(det_F, -2.0 / dim) * C) - dim -
            std::log(determinant(Q_t))) *
           df_mu_v_dH;
      B += 0.5 * this->get_mu_0() * this->get_mu_r() * det_F *
           dH_dot_C_inv_dot_H_dH;

      // Magnetostatic tangent: B = - pdpsi_dH
      // Now we treat Q_t = Q_t(C) --> B = B (C, Q(C))
      BB = -(0.5 * this->get_mu_e() * (tr_C - dim - 2.0 * log_det_F)) *
           d2f_mu_e_dH_dH;
      BB -= (0.5 * this->get_mu_v()) *
            (Q_t * (std::pow(det_F, -2.0 / dim) * C) - dim -
             std::log(determinant(Q_t))) *
            d2f_mu_v_dH_dH;
      BB += 0.5 * this->get_mu_0() * this->get_mu_r() * det_F *
            d2H_dot_C_inv_dot_H_dH_dH;

      // Piola-Kirchhoff stress: S = 2*pdpsi_dC
      S = 2.0 * (0.5 * this->get_mu_e() * f_mu_e) *                         //
            (d_tr_C_dC - 2.0 * dlog_det_F_dC)                               //
          + 2.0 * this->get_lambda_e() * (2.0 * log_det_F * dlog_det_F_dC); //
      S += 2.0 * (0.5 * this->get_mu_v() * f_mu_v) *
           ((Q_t * C) *
              ((-2.0 / dim) * std::pow(det_F, -2.0 / dim - 1.0) * ddet_F_dC) +
            std::pow(det_F, -2.0 / dim) * Q_t);                // dC/dC = II
      S -= 2.0 * (0.5 * this->get_mu_0() * this->get_mu_r()) * //
           (H_dot_C_inv_dot_H * ddet_F_dC                      //
            + det_F * dH_dot_C_inv_dot_H_dC);                  //

      // Magnetoelastic coupling tangent: PP = -dS/dH
      // Now we treat Q_t = Q_t(C) --> S = S(C, Q(C))
      PP = -2.0 * (0.5 * this->get_mu_e()) *
           outer_product(Tensor<2, dim>(d_tr_C_dC - 2.0 * dlog_det_F_dC),
                         df_mu_e_dH);
      PP -= 2.0 * (0.5 * this->get_mu_v()) *
            outer_product(Tensor<2, dim>((Q_t * C) *
                                           ((-2.0 / dim) *
                                            std::pow(det_F, -2.0 / dim - 1.0) *
                                            ddet_F_dC) +
                                         std::pow(det_F, -2.0 / dim) * Q_t),
                          df_mu_v_dH);
      PP += 2.0 * (0.5 * this->get_mu_0() * this->get_mu_r()) *
            (outer_product(Tensor<2, dim>(ddet_F_dC), dH_dot_C_inv_dot_H_dH) +
             det_F * d2H_dot_C_inv_dot_H_dC_dH);

      // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
      // Now we treat Q_t = Q_t(C) --> S = S(C, Q(C))
      HH =
        4.0 * (0.5 * this->get_mu_e() * f_mu_e) * (-2.0 * d2log_det_F_dC_dC) //
        + 4.0 * this->get_lambda_e() *                                       //
            (2.0 * outer_product(dlog_det_F_dC, dlog_det_F_dC)               //
             + 2.0 * log_det_F * d2log_det_F_dC_dC);                         //
      HH += 4.0 * (0.5 * this->get_mu_v() * f_mu_v) *
            (outer_product((-2.0 / dim) * std::pow(det_F, -2.0 / dim - 1.0) *
                             ddet_F_dC,
                           C * dQ_t_dC + Q_t) +
             (Q_t * C) *
               (outer_product(ddet_F_dC,
                              (-2.0 / dim) * (-2.0 / dim - 1.0) *
                                std::pow(det_F, -2.0 / dim - 2.0) * ddet_F_dC) +
                ((-2.0 / dim) * std::pow(det_F, -2.0 / dim - 1.0) *
                 d2det_F_dC_dC)) +
             outer_product(Q_t,
                           (-2.0 / dim) * std::pow(det_F, -2.0 / dim - 1.0) *
                             ddet_F_dC) +
             std::pow(det_F, -2.0 / dim) * dQ_t_dC);
      HH -= 4.0 * (0.5 * this->get_mu_0() * this->get_mu_r()) * //
            (H_dot_C_inv_dot_H * d2det_F_dC_dC                  //
             + outer_product(ddet_F_dC, dH_dot_C_inv_dot_H_dC)  //
             + outer_product(dH_dot_C_inv_dot_H_dC, ddet_F_dC)  //
             + det_F * d2H_dot_C_inv_dot_H_dC_dC);              //


      // Now that we're done using all of those temporary variables stored
      // in our cache, we can clear it out to free up some memory.
      cache.reset();
    }

    // Free energy
    template <int dim>
    double Magnetoviscoelastic_Constitutive_Law<dim>::get_psi() const
    {
      return psi;
    }

    // Magnetic induction: B = -dpsi/dH
    template <int dim>
    Tensor<1, dim> Magnetoviscoelastic_Constitutive_Law<dim>::get_B() const
    {
      return B;
    }

    // Piola-Kirchhoff stress tensor: S = 2*dpsi/dC
    template <int dim>
    SymmetricTensor<2, dim>
    Magnetoviscoelastic_Constitutive_Law<dim>::get_S() const
    {
      return S;
    }

    // Magnetostatic tangent: BB = dB/dH = - d2psi/dH.dH
    template <int dim>
    SymmetricTensor<2, dim>
    Magnetoviscoelastic_Constitutive_Law<dim>::get_BB() const
    {
      return BB;
    }

    // Magnetoelastic coupling tangent: PP = -dS/dH = -d/dH(2*dpsi/dC)
    template <int dim>
    Tensor<3, dim> Magnetoviscoelastic_Constitutive_Law<dim>::get_PP() const
    {
      return PP;
    }

    // Material elastic tangent: HH = 2*dS/dC = 4*d2psi/dC.dC
    template <int dim>
    SymmetricTensor<4, dim>
    Magnetoviscoelastic_Constitutive_Law<dim>::get_HH() const
    {
      return HH;
    }

    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law<dim>::update_end_of_timestep()
    {
      Q_t1 = Q_t;
    };


    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law<dim>::set_primary_variables(
      const Tensor<1, dim> &         H,
      const SymmetricTensor<2, dim> &C) const
    {
      // Set value for H
      const std::string name_H("H");
      Assert(!cache.stores_object_with_name(name_H),
             ExcMessage(
               "The primary variable has already been added to the cache."));
      cache.add_unique_copy(name_H, H);

      // Set value for C
      const std::string name_C("C");
      Assert(!cache.stores_object_with_name(name_C),
             ExcMessage(
               "The primary variable has already been added to the cache."));
      cache.add_unique_copy(name_C, C);
    }


    template <int dim>
    void Magnetoviscoelastic_Constitutive_Law<dim>::update_internal_variable(
      const DiscreteTime &time)
    {
      const double delta_t = time.get_previous_step_size();
      // Evolution law: See Linder2011a eq. 41
      // Discretizing in time (BDF 1) gives us this expression,
      // i.e., @cite Linder2011a eq. 54
      // Note: std::pow(det_F, 2.0 / dim) * C_inv == C_bar_inv
      Q_t = (1.0 / (1.0 + delta_t / this->get_tau_v())) *
            (Q_t1 + (delta_t / this->get_tau_v()) *
                      std::pow(get_det_F(), 2.0 / dim) * get_C_inv());
    }



    // =========
    // Primary variables

    template <int dim>
    const Tensor<1, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_H() const
    {
      const std::string name("H");
      Assert(cache.stores_object_with_name(name),
             ExcMessage("Primary variables must be added to the cache."));
      return cache.template get_object_with_name<Tensor<1, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_C() const
    {
      const std::string name("C");
      Assert(cache.stores_object_with_name(name),
             ExcMessage("Primary variables must be added to the cache."));
      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    // =========
    // Scaling function and its derivatives

    template <int dim>
    double
    Magnetoviscoelastic_Constitutive_Law<dim>::get_two_h_dot_h_div_h_sat_squ(
      const double mu_h_sat) const
    {
      const Tensor<1, dim> &H = get_H();
      return (2.0 * H * H) / (mu_h_sat * mu_h_sat);
    };

    template <int dim>
    double Magnetoviscoelastic_Constitutive_Law<
      dim>::get_tanh_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const
    {
      return std::tanh(get_two_h_dot_h_div_h_sat_squ(mu_h_sat));
    };

    // A scaling function that will cause the shear modulus
    // to change (increase) under the influence of a magnetic
    // field.
    template <int dim>
    double Magnetoviscoelastic_Constitutive_Law<dim>::get_f_mu(
      const double mu,
      const double mu_inf,
      const double mu_h_sat) const
    {
      return 1.0 +
             (mu_inf / mu - 1.0) * get_tanh_two_h_dot_h_div_h_sat_squ(mu_h_sat);
    };

    // First derivative of scaling function
    template <int dim>
    double Magnetoviscoelastic_Constitutive_Law<
      dim>::get_dtanh_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const
    {
      return std::pow(1.0 / std::cosh(get_two_h_dot_h_div_h_sat_squ(mu_h_sat)),
                      2.0); // d/dx [tanh(x)] = sech^2(x)
    };

    template <int dim>
    Tensor<1, dim> Magnetoviscoelastic_Constitutive_Law<
      dim>::get_dtwo_h_dot_h_div_h_sat_squ_dH(const double mu_h_sat) const
    {
      return 2.0 * 2.0 / (mu_h_sat * mu_h_sat) * get_H();
    };

    template <int dim>
    Tensor<1, dim> Magnetoviscoelastic_Constitutive_Law<dim>::get_df_mu_dH(
      const double mu,
      const double mu_inf,
      const double mu_h_sat) const
    {
      return (mu_inf / mu - 1.0) *
             (get_dtanh_two_h_dot_h_div_h_sat_squ(mu_h_sat) *
              get_dtwo_h_dot_h_div_h_sat_squ_dH(mu_h_sat));
    };

    // Second derivative of scaling function
    template <int dim>
    double Magnetoviscoelastic_Constitutive_Law<
      dim>::get_d2tanh_two_h_dot_h_div_h_sat_squ(const double mu_h_sat) const
    {
      return -2.0 * get_tanh_two_h_dot_h_div_h_sat_squ(mu_h_sat) *
             get_dtanh_two_h_dot_h_div_h_sat_squ(mu_h_sat);
    };

    template <int dim>
    SymmetricTensor<2, dim> Magnetoviscoelastic_Constitutive_Law<
      dim>::get_d2two_h_dot_h_div_h_sat_squ_dH_dH(const double mu_h_sat) const
    {
      return 2.0 * 2.0 / (mu_h_sat * mu_h_sat) *
             Physics::Elasticity::StandardTensors<dim>::I;
    };

    template <int dim>
    SymmetricTensor<2, dim>
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d2f_mu_dH_dH(
      const double mu,
      const double mu_inf,
      const double mu_h_sat) const
    {
      return (mu_inf / mu - 1.0) *
             (get_d2tanh_two_h_dot_h_div_h_sat_squ(mu_h_sat) *
                symmetrize(
                  outer_product(get_dtwo_h_dot_h_div_h_sat_squ_dH(mu_h_sat),
                                get_dtwo_h_dot_h_div_h_sat_squ_dH(mu_h_sat))) +
              get_dtanh_two_h_dot_h_div_h_sat_squ(mu_h_sat) *
                get_d2two_h_dot_h_div_h_sat_squ_dH_dH(mu_h_sat));
    };

    // =========
    // Intermediate values directly attained from primary variables

    template <int dim>
    const double &Magnetoviscoelastic_Constitutive_Law<dim>::get_det_F() const
    {
      const std::string name("det_F");

      // If the cache does not already store the value that
      // we're looking for, then we quickly calculate it, store
      // it in the cache and return the value just stored in
      // the cache (rather than the intermediate value.
      // That way we can return it as a reference and avoid
      // copying the object.
      if (cache.stores_object_with_name(name) == false)
        {
          const double det_F = std::sqrt(determinant(get_C()));
          AssertThrow(det_F > 0.0,
                      ExcMessage("Volumetric Jacobian must be positive."));
          cache.add_unique_copy(name, det_F);
        }

      return cache.template get_object_with_name<double>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_C_inv() const
    {
      const std::string name("C_inv");
      if (cache.stores_object_with_name(name) == false)
        {
          cache.add_unique_copy(name, invert(get_C()));
        }

      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    // =========

    template <int dim>
    const double &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_log_det_F() const
    {
      const std::string name("log(det_F)");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, std::log(get_det_F()));

      return cache.template get_object_with_name<double>(name);
    }

    template <int dim>
    const double &Magnetoviscoelastic_Constitutive_Law<dim>::get_trace_C() const
    {
      const std::string name("trace(C)");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, trace(get_C()));

      return cache.template get_object_with_name<double>(name);
    }

    template <int dim>
    const Tensor<1, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_C_inv_dot_H() const
    {
      const std::string name("C_inv_dot_H");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, get_C_inv() * get_H());

      return cache.template get_object_with_name<Tensor<1, dim>>(name);
    }

    template <int dim>
    const double &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_H_dot_C_inv_dot_H() const
    {
      const std::string name("H_dot_C_inv_dot_H");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, get_H() * get_C_inv_dot_H());

      return cache.template get_object_with_name<double>(name);
    }

    // =========
    // First derivatives

    // Derivative of internal variable wrt. field variables
    template <int dim>
    const SymmetricTensor<4, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_dQ_t_dC(
      const DiscreteTime &time) const
    {
      const std::string name("dQ_t_dC");
      if (cache.stores_object_with_name(name) == false)
        {
          const double  delta_t = time.get_previous_step_size();
          const double &det_F   = get_det_F();

          const SymmetricTensor<4, dim> dQ_t_dC =
            (1.0 / (1.0 + delta_t / this->get_tau_v())) *
            (delta_t / this->get_tau_v()) *
            ((2.0 / dim) * std::pow(det_F, 2.0 / dim - 1.0) *
               outer_product(get_C_inv(), get_ddet_F_dC()) +
             std::pow(det_F, 2.0 / dim) * get_dC_inv_dC());

          cache.add_unique_copy(name, dQ_t_dC);
        }

      return cache.template get_object_with_name<SymmetricTensor<4, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<4, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_dC_inv_dC() const
    {
      const std::string name("dC_inv_dC");
      if (cache.stores_object_with_name(name) == false)
        {
          const SymmetricTensor<2, dim> &C_inv = get_C_inv();
          SymmetricTensor<4, dim>        dC_inv_dC;

          for (unsigned int A = 0; A < dim; ++A)
            for (unsigned int B = A; B < dim; ++B)
              for (unsigned int C = 0; C < dim; ++C)
                for (unsigned int D = C; D < dim; ++D)
                  dC_inv_dC[A][B][C][D] -=               //
                    0.5 * (C_inv[A][C] * C_inv[B][D]     //
                           + C_inv[A][D] * C_inv[B][C]); //

          cache.add_unique_copy(name, dC_inv_dC);
        }

      return cache.template get_object_with_name<SymmetricTensor<4, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d_tr_C_dC() const
    {
      const std::string name("d_tr_C_dC");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name,
                              Physics::Elasticity::StandardTensors<dim>::I);

      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_ddet_F_dC() const
    {
      const std::string name("ddet_F_dC");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, 0.5 * get_det_F() * get_C_inv());

      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_dlog_det_F_dC() const
    {
      const std::string name("dlog_det_F_dC");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, 0.5 * get_C_inv());

      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    template <int dim>
    const Tensor<1, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_dH_dot_C_inv_dot_H_dH() const
    {
      const std::string name("dH_dot_C_inv_dot_H_dH");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, 2.0 * get_C_inv_dot_H());

      return cache.template get_object_with_name<Tensor<1, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_dH_dot_C_inv_dot_H_dC() const
    {
      const std::string name("dH_dot_C_inv_dot_H_dC");
      if (cache.stores_object_with_name(name) == false)
        {
          const Tensor<1, dim> C_inv_dot_H = get_C_inv_dot_H();
          cache.add_unique_copy(
            name, -symmetrize(outer_product(C_inv_dot_H, C_inv_dot_H)));
        }

      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    // =========
    // Second derivatives

    template <int dim>
    const SymmetricTensor<4, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d2log_det_F_dC_dC() const
    {
      const std::string name("d2log_det_F_dC_dC");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, 0.5 * get_dC_inv_dC());

      return cache.template get_object_with_name<SymmetricTensor<4, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<4, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d2det_F_dC_dC() const
    {
      const std::string name("d2det_F_dC_dC");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name,
                              0.5 *
                                (outer_product(get_C_inv(), get_ddet_F_dC()) +
                                 get_det_F() * get_dC_inv_dC()));

      return cache.template get_object_with_name<SymmetricTensor<4, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<2, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d2H_dot_C_inv_dot_H_dH_dH()
      const
    {
      const std::string name("d2H_dot_C_inv_dot_H_dH_dH");
      if (cache.stores_object_with_name(name) == false)
        cache.add_unique_copy(name, 2.0 * get_C_inv());

      return cache.template get_object_with_name<SymmetricTensor<2, dim>>(name);
    }

    template <int dim>
    const Tensor<3, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d2H_dot_C_inv_dot_H_dC_dH()
      const
    {
      const std::string name("d2H_dot_C_inv_dot_H_dC_dH");
      if (cache.stores_object_with_name(name) == false)
        {
          const Tensor<1, dim> &         C_inv_dot_H = get_C_inv_dot_H();
          const SymmetricTensor<2, dim> &C_inv       = get_C_inv();

          Tensor<3, dim> d2H_dot_C_inv_dot_H_dC_dH;
          for (unsigned int A = 0; A < dim; ++A)
            for (unsigned int B = 0; B < dim; ++B)
              for (unsigned int C = 0; C < dim; ++C)
                d2H_dot_C_inv_dot_H_dC_dH[A][B][C] -=
                  C_inv[A][C] * C_inv_dot_H[B] + //
                  C_inv_dot_H[A] * C_inv[B][C];  //

          cache.add_unique_copy(name, d2H_dot_C_inv_dot_H_dC_dH);
        }

      return cache.template get_object_with_name<Tensor<3, dim>>(name);
    }

    template <int dim>
    const SymmetricTensor<4, dim> &
    Magnetoviscoelastic_Constitutive_Law<dim>::get_d2H_dot_C_inv_dot_H_dC_dC()
      const
    {
      const std::string name("d2H_dot_C_inv_dot_H_dC_dC");
      if (cache.stores_object_with_name(name) == false)
        {
          const Tensor<1, dim> &         C_inv_dot_H = get_C_inv_dot_H();
          const SymmetricTensor<2, dim> &C_inv       = get_C_inv();

          SymmetricTensor<4, dim> d2H_dot_C_inv_dot_H_dC_dC;
          for (unsigned int A = 0; A < dim; ++A)
            for (unsigned int B = A; B < dim; ++B)
              for (unsigned int C = 0; C < dim; ++C)
                for (unsigned int D = C; D < dim; ++D)
                  d2H_dot_C_inv_dot_H_dC_dC[A][B][C][D] +=
                    0.5 * (C_inv_dot_H[A] * C_inv_dot_H[C] * C_inv[B][D] +
                           C_inv_dot_H[A] * C_inv_dot_H[D] * C_inv[B][C] +
                           C_inv_dot_H[B] * C_inv_dot_H[C] * C_inv[A][D] +
                           C_inv_dot_H[B] * C_inv_dot_H[D] * C_inv[A][C]);

          cache.add_unique_copy(name, d2H_dot_C_inv_dot_H_dC_dC);
        }

      return cache.template get_object_with_name<SymmetricTensor<4, dim>>(name);
    }


    // @sect4{Rheological experiment parameters}

    class RheologicalExperimentParameters : public ParameterAcceptor
    {
    public:
      RheologicalExperimentParameters();

      // The dimensions of the rheological specimen that is to be simulated.
      // These, effectively, define the measurement point for our virtual
      // experiment.
      double sample_radius = 0.01;
      double sample_height = 0.001;

      // The three steady-state loading parameters are respectively
      // - the axial stretch,
      // - the shear strain amplitude, and
      // - the axial magnetic field strength.
      double lambda_2 = 0.95;
      double gamma_12 = 0.05;
      double H_2      = 60.0e3;

      // Moreover, the parameters for the time-dependent rheological
      // loading conditions are
      // - the loading cycle frequency,
      // - the number of load cycles, and
      // - the number of discrete timesteps per cycle.
      double       frequency         = 1.0 / (2.0 * numbers::PI);
      unsigned int n_cycles          = 5;
      unsigned int n_steps_per_cycle = 2500;

      // We also declare some self-explanatory parameters related to output
      // data generated for the experiments conducted with rate-dependent and
      // rate-independent materials.
      bool        output_data_to_file = true;
      std::string output_filename_rd =
        "experimental_results-rate_dependent.csv";
      std::string output_filename_ri =
        "experimental_results-rate_independent.csv";

      double start_time() const
      {
        return 0.0;
      }

      double end_time() const
      {
        return n_cycles / frequency;
      }

      double delta_t() const
      {
        return (end_time() - start_time()) / (n_steps_per_cycle * n_cycles);
      }

      bool print_status(const int step_number) const
      {
        return (step_number % (n_cycles * n_steps_per_cycle / 100)) == 0;
      }

      Tensor<1, 3> get_H(const double &time) const;
      Tensor<2, 3> get_F(const double &time) const;

      bool initialized = false;
    };

    RheologicalExperimentParameters::RheologicalExperimentParameters()
      : ParameterAcceptor("/Coupled Constitutive Laws/Rheological Experiment/")
    {
      add_parameter("Experimental sample radius", sample_radius);
      add_parameter("Experimental sample radius", sample_height);

      add_parameter("Axial stretch", lambda_2);
      add_parameter("Shear strain amplitude", gamma_12);
      add_parameter("Axial magnetic field strength", H_2);

      add_parameter("Frequency", frequency);
      add_parameter("Number of loading cycles", n_cycles);
      add_parameter("Discretisation for each cycle", n_steps_per_cycle);

      add_parameter("Output experimental results to file", output_data_to_file);
      add_parameter("Output file name (rate dependent constitutive law)",
                    output_filename_rd);
      add_parameter("Output file name (rate independent constitutive law)",
                    output_filename_ri);

      parse_parameters_call_back.connect([&]() -> void { initialized = true; });
    }

    Tensor<1, 3> RheologicalExperimentParameters::get_H(const double &) const
    {
      return Tensor<1, 3>({0.0, 0.0, H_2});
    }

    Tensor<2, 3>
    RheologicalExperimentParameters::get_F(const double &time) const
    {
      AssertThrow((sample_radius > 0.0 && sample_height > 0.0),
                  ExcMessage("Non-physical sample dimensions"));
      AssertThrow(lambda_2 > 0.0,
                  ExcMessage("Non-physical applied axial stretch"));

      const double sqrt_lambda_2     = std::sqrt(lambda_2);
      const double inv_sqrt_lambda_2 = 1.0 / sqrt_lambda_2;

      const double alpha_max =
        std::atan(std::tan(gamma_12) * sample_height /
                  sample_radius); // Small strain approximation
      const double A       = sample_radius * alpha_max;
      const double w       = 2.0 * numbers::PI * frequency; // rad /s
      const double gamma_t = A * std::sin(w * time);
      const double tau_t =
        gamma_t /
        (sample_radius * sample_height); // Torsion angle per unit length
      const double alpha_t = tau_t * lambda_2 * sample_height;

      Tensor<2, 3> F;
      F[0][0] = inv_sqrt_lambda_2 * std::cos(alpha_t);
      F[0][1] = -inv_sqrt_lambda_2 * std::sin(alpha_t);
      F[0][2] = -tau_t * sample_radius * sqrt_lambda_2 * std::sin(alpha_t);
      F[1][0] = inv_sqrt_lambda_2 * std::sin(alpha_t);
      F[1][1] = inv_sqrt_lambda_2 * std::cos(alpha_t);
      F[1][2] = tau_t * sample_radius * sqrt_lambda_2 * std::cos(alpha_t);
      F[2][0] = 0.0;
      F[2][1] = 0.0;
      F[2][2] = lambda_2;

      AssertThrow((F[0][0] > 0) && (F[1][1] > 0) && (F[2][2] > 0),
                  ExcMessage("Non-physical deformation gradient component."));
      AssertThrow(std::abs(determinant(F) - 1.0) < 1e-6,
                  ExcMessage("Volumetric Jacobian is not equal to unity."));

      return F;
    }


    // @sect4{Rheological experiment: Parallel plate rotational rheometer}

    template <int dim>
    void run_rheological_experiment(
      const RheologicalExperimentParameters &experimental_parameters,
      Coupled_Magnetomechanical_Constitutive_Law_Base<dim>
        &material_hand_calculated,
      Coupled_Magnetomechanical_Constitutive_Law_Base<dim>
        &               material_assisted_computation,
      TimerOutput &     timer,
      const std::string filename)
    {
      auto check_material_class_results =
        [](
          const Coupled_Magnetomechanical_Constitutive_Law_Base<dim> &to_verify,
          const Coupled_Magnetomechanical_Constitutive_Law_Base<dim> &blessed,
          const double tol = 1e-6) {
          (void)to_verify;
          (void)blessed;
          (void)tol;

          // Compare free energy
          Assert(std::abs(blessed.get_psi() - to_verify.get_psi()) < tol,
                 ExcMessage("No match for psi. Error: " +
                            Utilities::to_string(std::abs(
                              blessed.get_psi() - to_verify.get_psi()))));

          // Compare first derivatives of free energy
          Assert((blessed.get_B() - to_verify.get_B()).norm() < tol,
                 ExcMessage("No match for B. Error: " +
                            Utilities::to_string(
                              (blessed.get_B() - to_verify.get_B()).norm())));
          Assert((blessed.get_S() - to_verify.get_S()).norm() < tol,
                 ExcMessage("No match for S. Error: " +
                            Utilities::to_string(
                              (blessed.get_S() - to_verify.get_S()).norm())));

          // Compare second derivatives of free energy
          Assert((blessed.get_BB() - to_verify.get_BB()).norm() < tol,
                 ExcMessage("No match for BB. Error: " +
                            Utilities::to_string(
                              (blessed.get_BB() - to_verify.get_BB()).norm())));
          Assert((blessed.get_PP() - to_verify.get_PP()).norm() < tol,
                 ExcMessage("No match for PP. Error: " +
                            Utilities::to_string(
                              (blessed.get_PP() - to_verify.get_PP()).norm())));
          Assert((blessed.get_HH() - to_verify.get_HH()).norm() < tol,
                 ExcMessage("No match for HH. Error: " +
                            Utilities::to_string(
                              (blessed.get_HH() - to_verify.get_HH()).norm())));
        };

      std::ostringstream stream;
      stream
        << "Time;Axial magnetic field strength [A/m];Axial magnetic induction [T];Shear strain [%];Shear stress [Pa]\n";

      for (DiscreteTime time(experimental_parameters.start_time(),
                             experimental_parameters.end_time() +
                               experimental_parameters.delta_t(),
                             experimental_parameters.delta_t());
           time.is_at_end() == false;
           time.advance_time())
        {
          if (experimental_parameters.print_status(time.get_step_number()))
            std::cout << "Timestep = " << time.get_step_number()
                      << " @ time = " << time.get_current_time() << "s."
                      << std::endl;

          const Tensor<1, dim> H =
            experimental_parameters.get_H(time.get_current_time());
          const Tensor<2, dim> F =
            experimental_parameters.get_F(time.get_current_time());
          const SymmetricTensor<2, dim> C =
            Physics::Elasticity::Kinematics::C(F);

          {
            TimerOutput::Scope timer_section(timer, "Hand calculated");
            material_hand_calculated.update_internal_data(H, C, time);
            material_hand_calculated.update_end_of_timestep();
          }

          if (experimental_parameters.output_data_to_file)
            {
              // Collect some results to post-process.
              // All quantities are in the current configuration.
              const Tensor<1, dim> h =
                Physics::Transformations::Covariant::push_forward(H, F);
              const Tensor<1, dim> b =
                Physics::Transformations::Piola::push_forward(
                  material_hand_calculated.get_B(), F);
              const SymmetricTensor<2, dim> sigma =
                Physics::Transformations::Piola::push_forward(
                  material_hand_calculated.get_S(), F);
              stream << time.get_current_time() << ";" << h[2] << ";" << b[2]
                     << ";" << F[1][2] * 100.0 << ";" << sigma[1][2] << "\n";
            }

          {
            TimerOutput::Scope timer_section(timer, "Assisted computation");
            material_assisted_computation.update_internal_data(H, C, time);
            material_assisted_computation.update_end_of_timestep();
          }

          check_material_class_results(material_hand_calculated,
                                       material_assisted_computation);
        }

      // Output strain-stress history to file
      if (experimental_parameters.output_data_to_file)
        {
          std::ofstream output(filename);
          output << stream.str();
          output.close();
        }
    };

  } // namespace CoupledConstitutiveLaws

} // namespace Step71


// @sect3{The main() function}

int main(int argc, char *argv[])
{
  {
    using namespace Step71::SimpleExample;

    // Choose some values at which to evaluate the function
    const double x = 1.23;
    const double y = 0.91;

    std::cout << "Simple example using automatic differentiation..."
              << std::endl;
    run_and_verify_ad(x, y);
    std::cout << "... all calculations are correct!" << std::endl;

    std::cout << "Simple example using symbolic differentiation." << std::endl;
    run_and_verify_sd(x, y);
    std::cout << "... all calculations are correct!" << std::endl;
  }

  {
    using namespace dealii;
    using namespace dealii::Differentiation;
    using namespace Step71::CoupledConstitutiveLaws;

    constexpr unsigned int dim = 3;

    const ConstitutiveParameters          constitutive_parameters;
    const RheologicalExperimentParameters experimental_parameters;

    // Rate-independent constitutive law
    {
      TimerOutput timer(std::cout,
                        TimerOutput::summary,
                        TimerOutput::wall_times);
      std::cout
        << "Coupled magnetoelastic constitutive law using automatic differentiation."
        << std::endl;

      constexpr AD::NumberTypes ADTypeCode = AD::NumberTypes::sacado_dfad_dfad;

      Magnetoelastic_Constitutive_Law<dim> material(constitutive_parameters);
      Magnetoelastic_Constitutive_Law_AD<dim, ADTypeCode> material_AD(
        constitutive_parameters);

      run_rheological_experiment(experimental_parameters,
                                 material,
                                 material_AD,
                                 timer,
                                 experimental_parameters.output_filename_ri);

      std::cout << "... all calculations are correct!" << std::endl;
    }

    // Rate-dependent constitutive law
    {
      TimerOutput timer(std::cout,
                        TimerOutput::summary,
                        TimerOutput::wall_times);
      std::cout
        << "Coupled magneto-viscoelastic constitutive law using symbolic differentiation."
        << std::endl;

#ifdef DEAL_II_SYMENGINE_WITH_LLVM
      std::cout << "Using LLVM optimizer." << std::endl;
      constexpr SD::OptimizerType     optimizer_type = SD::OptimizerType::llvm;
      constexpr SD::OptimizationFlags optimization_flags =
        SD::OptimizationFlags::optimize_all;
#else
      std::cout << "Using lambda optimizer." << std::endl;
      constexpr SD::OptimizerType optimizer_type = SD::OptimizerType::lambda;
      constexpr SD::OptimizationFlags optimization_flags =
        SD::OptimizationFlags::optimize_cse;
#endif

      Magnetoviscoelastic_Constitutive_Law<dim> material(
        constitutive_parameters);
      // Note: Want to invoke optimizer only once! So we do it in the
      // class constructor.
      timer.enter_subsection("Initialize symbolic CL");
      Magnetoviscoelastic_Constitutive_Law_SD<dim> material_SD(
        constitutive_parameters, optimizer_type, optimization_flags);
      timer.leave_subsection();

      run_rheological_experiment(experimental_parameters,
                                 material,
                                 material_SD,
                                 timer,
                                 experimental_parameters.output_filename_rd);

      std::cout << "... all calculations are correct!" << std::endl;
    }

    std::string parameter_file;
    if (argc > 1)
      parameter_file = argv[1];
    else
      parameter_file = "parameters.prm";
    ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
  }

  return 0;
}
