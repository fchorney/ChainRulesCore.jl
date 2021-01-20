# Design Notes: Why can you change the primal?
These design notes are to help you understand why ChainRules allows the primal computation, to be changed.
We will focus this discussion on reverse mode and `rrule`, though the same also applies to forwards mode and `frule`.
In fact in forwards mode it has particular uses for efficiently calculate the pushfoward of a differential equation solve via expanding the system of equations to also include the derivatives, and solving all at ones.
In forwards mode it related to the fusing of `frule` and `pushforward`.
In reverse-mode we can focus on the the distinct primal and gradient passes.


Let's imagine a different system for rules, one that doesn't let you do this.
This system is what a lot of AD system have --- it is what [Nabla.jl](https://github.com/invenia/Nabla.jl/)[^1] had originally.
We will have a primal (i.e. forwards) pass that directly executes the primal function, and just records, it's _inputs_ and it's _output_ (as as well as the _primal function_ itself) onto the tape.[^2].
Then during the gradient (i.e. reverse) pass it has a function which receives those records from the tape, plus the sensitivity of the output sensitivity, and gives back the sensitivity of the input.
We will call this function `pullback_at`, as it pulls back the sensitivity at a given primal point.
To make this concrete:
```julia
y = f(x)  # primal program
x̄ = pullback_at(f, x, y, ȳ)
```

Let's write one:
```julia
y = sin(x)
pullback_at(::typeof(sin), x, y, ȳ) = ȳ * cos(x)
```

Great. So far so good.
As a short exercise the reader might like to implement the one for the [logistic sigmoid](https://en.wikipedia.org/wiki/Logistic_function#Derivative).
It also works without issue.


Now lets consider why we implement `rrules` like this in the first-place.
One key reason, [^3] is to allow us to insert our domain knowledge to do better than the AD would do just by breaking everything down into `+` and `*` etc.
What insights do we have about `sin` and `cos`?
Here is one:
```julia
julia> @btime (sin(x); cos(x)) setup=(x=rand());
  6.927 ns (0 allocations: 0 bytes)

julia> @btime sincos(x) setup=(x=rand());
  6.028 ns (0 allocations: 0 bytes)
```
It is \~15%[^4] faster to compute `sin` and `cos` at the same time via `sincos` than it is to compute them one after the other.









[^1]:
    I am not just picking on Nabla randomly.
    Many of the core developers of ChainRules worked on Nabla prior.
    It's a good AD, but ChainRules incorporates lessons learned from working on Nabla.

[^2]: which may be an explicit tape, or an implicit tape that is actually incorporated into generated code (ala Zygote)

[^3]:
    Another key reason is if the operations is a primitive that is not defined in terms of more basic operations.
    In many languages this is the case for `sin`; where the actual implementation is in some separate `libm.so`.
    But actually [`sin` in Julia is defined in terms of a polynomial](https://github.com/JuliaLang/julia/blob/caeaceff8af97565334f35309d812566183ec687/base/special/trig.jl).
    It's fairly vanilla julia code.
    It shouldn't be too hard for an AD that only knows about basic operations like `+` and `*` to AD through it.
    Though that will incur something that looks a lot like truncation error (in apparent violation of Griewank and Walther's 0th Rule of AD).
    In anycase, that is another discussion, for another day.

[^4]: Sure, this is small-fries and depending on julia version might just get solved by the optimizer, but go with it for the sake of example.