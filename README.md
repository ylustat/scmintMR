# scmintMR (under development)

`scmintMR` is a package implementing the integrative multi-context Mendelian randomization method for identifying risk genes across cell types

Installation
============

To install the development version of the `mintMR` package, please load the `devtools` package first. 

```
library(devtools)
install_github("ylustat/MVL")
install_github("ylustat/scmintMR")
```

To use the Deep learning algorithms and missing value imputation characteristics of mintMR, please do following steps first before applying the model to analyze data.

```R
library(reticulate)
library(MVL)
library(scmintMR)

python_path <- py_discover_config()$python
use_python(python_path)
source_python(paste0(system.file("python", package = "MVL", mustWork = TRUE),"/DeepCCA_base.py"))
```



### Additional notes

If you encounter the following messages when installing this package on a server without admin access, please see the solutions below:

- Please set `options(buildtools.check = function(action) TRUE )` before installation.

- If the error message shows

  ```R
  ERROR: 'configure' exists but is not executable -- see the 'R Installation and Administration Manual'
  ```

  Please follow instructions on this [page](https://vsoch.github.io/2013/install-r-packages-that-require-compilation-on-linux-without-sudo/).

