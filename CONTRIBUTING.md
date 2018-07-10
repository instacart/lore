### General guidelines and philosophy for contribution
* Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code. See the Release Checklist below.
* Breaking changes will not be accepted until a major version release.

### Test locally
CI is run for all PR's. Contributions should be compatible with recent versions of Python 2 & 3. To run tests against a specific version of python:

```bash
$ lore test
$ LORE_PYTHON_VERSION=3.6.5 lore test
$ LORE_PYTHON_VERSION=2.7.15 lore test -s tests.unit.test_encoders.TestUniform.test_cardinality
```

You may need to allow requirements.txt to be recalculated when building different virtualenvs for python 2 and 3.
```bash
$ git checkout -- requirements.txt
```

Install a local version of lore in your project's lore env:

```bash
$ git clone https://github.com/instacart/lore ~/repos/lore
$ cd my_project
$ lore pip install -e ~/repos/lore
$ lore test
```

### Release Checklist:
* Did you add any required properties to Model/Estimator/Pipeline or other Base classes? You need to provide default values for serialized objects during deserialization.
* Did you add any new modules? You need to specify them in setup.py: packages.
* Did you add any new dependencies? Do not add them to setup.py. Instead add them in lore/dependencies.py, and require them only in modules that need it.

### Python coding style
Changes should conform to Google Python Style Guide, except feel free to exceed 80 char line limit.
Keep single logical statements on a single line, and use descriptive names. Underscores for functions and variables, camelcase for classes, capitalized underscored constants. In general, new code should follow the style of the existing code closest to it.

Do not fall prey to the 80 char line length limit. It leads to short, bad names like `q`, `tmp`, `xrt()`. It causes excessive declaration of single use temporary variables with those bad names, that chop logical statements into incoherant expressions. It discourages the use of named function arguments. It pollutes the global namespace by encouraging `from x import Y`, or worse `import package as pk`. It leads to poorly readable line wrapping of function arguments with _insignificant_ whitespace in a whitespace _significant_ language. It makes formatting user facing strings more error prone around whitespace. It breaks urls in docstrings. It costs developers time to format code. The argument for greater readability does not bear out in practice.

Use pylint to check your Python changes. To install pylint and retrieve Lore's custom style definition:
```bash
$ pip install pylint
$ wget -O /tmp/pylintrc https://raw.githubusercontent.com/instacart/lore/master/pylintrc
```
To check a file with pylint:
```bash
$ pylint myfile.py
```
