Welcome to the EasyML library! EasyML is a set of classic machine learning algorithms written in C++. It extensively uses the [Armadillo](https://arma.sourceforge.net/) library which in turn uses the LAPACK library to work with vectors and matrices. [Boost](https://www.boost.org/) is also used, for example, to obtain object's human-readable type during runtime and to work with probability distributions. Be sure to install them first.

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

You can browse the full classes and files doxygen documentation in the [docs/html](https://github.com/andrewha/ezml/tree/main/docs)

---
Libraries, tools, and OS

C++ | Armadillo | LAPACK | Boost | Bash | Linux
----|-----------|--------|-------|------|------
<img src="https://github.com/devicons/devicon/blob/master/icons/cplusplus/cplusplus-original.svg" alt="C++" width="75"/> | <img src="https://gitlab.com/uploads/-/system/project/avatar/6604173/armadillo_logo2.png" alt="Armadillo" width="75"/> | <img src="https://github.com/Reference-LAPACK/lapack/blob/master/DOCS/lapack.png" alt="LAPACK" width="75"/> | <img src="https://github.com/boostorg/boost/blob/master/boost.png" alt="Boost" width="75"/> | <img src="https://github.com/devicons/devicon/blob/master/icons/bash/bash-original.svg" alt="Bash" width="75"/> | <img src="https://github.com/devicons/devicon/blob/master/icons/linux/linux-original.svg" alt="Linux" width="75"/>
