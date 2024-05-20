Welcome to the EasyML library! EasyML is a set of classic machine learning algorithms written in C++. It extensively uses the [Armadillo](https://arma.sourceforge.net/) library which in turn uses the LAPACK library to work with vectors and matrices. [Boost](https://www.boost.org/) is also used, for example, to obtain object's human-readable type during runtime and to work with distributions. Be sure to install them first.

Supported models and solvers:
- Linear Regression
  - Ordinary Least Squares
  - QR-decomposition
  - Derivative-based: Gradient Descent, Newton (single feature only)
- Logistic Regression
  - Derivative-based: Gradient Descent, Newton (single feature only)
- Autoregressive AR(p)
  - Ordinary Least Squares
  - QR-decomposition
  - Derivative-based: Gradient Descent, Newton (single lag only)

Supported transformers and extractors:
- Standard scaler ($z$-score transformation)
- Time series (extract features and target from process)

More models and possibly transformers to be implemented in future versions.

---
Installation for Linux

* `> ./build_lib.sh`. The static library will be built into `./bin/static/libezml.a`
*  You can either copy it along with headers to your default location, or just keep it in your local project.

---
Usage Examples

To built any of the example files located in [examples](examples):
* `> cd examples`
* `> ./build_example.sh <filename without extension>`, i.e. `./build_example.sh linreg_single_toy`. The executable will be built into `./build/`.

---
Documentation

[Full classes and files documentation](https://ezmldocs-1-z1750187.deta.app)
