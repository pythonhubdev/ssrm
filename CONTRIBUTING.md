# Contributing

You can clone the repository from the command-line:

```console
git clone git@github.com:optimizely/ssrm.git
```

> **NOTE:** Feel free to use GUI-based git clients, like [GitHub Desktop](https://desktop.github.com/) or an IDE integration, but this documentation will only cover CLI-usage of git.

## Setting Up a Development Environment

### Virtualenv

We recommend that you use an isolated virtual environment to install and run the code in this repo (See: [virtualenv](https://pypi.org/project/virtualenv/) and [pyenv](https://github.com/pyenv/pyenv))


### Install dev dependencies

Now, we are ready to install the core dependencies, as well as other dev-specific tools to run tests, build the docs, etc:

    make install-dev

> **Tip:** have a look in the [`Makefile`](/Makefile) to learn more about what this make recipe does!

### Running Jupyter

With the virtualenv active, start the Jupyter server with:

    jupyter lab


You should be all set!
There's more helpful info below; so please read to the end.

If you run into any issues along the way, please open an issue on the GitHub repo or reach out to team members directly.

Happy coding!

## Tests

We use [pytest](https://docs.pytest.org/en/latest/) for tests and
[flake8](http://flake8.pycqa.org/en/latest/) for linting.

### Running Tests

-   `make check` to run all checks
-   `make test` to run unit tests

## Code Style

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

The code style used in this repository is fairly standard:
-   [black](https://black.readthedocs.io/en/stable/) as a code formatter and
-   [isort](https://github.com/timothycrosley/isort) for consistency.

We use the [`pre-commit`](https://pre-commit.com/) to run these tools as the name suggests.

If you used `make install-dev` to set up your development environment, git-hooks should have been
installed, but you can also invoke pre-commit manually yourself.

> **TIP:** It's highly-recommended to setup an [editor integration](https://black.readthedocs.io/en/stable/editor_integration.html) for `black`!

## Documentation

### Docstring format

❗️ This repository has adopted the [numpydoc docstring format][numpydoc] ❗️

All contributors should be familiar with [numpydoc][] as well as the docstring conventions
of [PEP 257](https://www.python.org/dev/peps/pep-0257/).

If your change introduces any user-facing modifications, please:

1. Add or update docstrings.
2. Update [release notes](#release-notes).

**See also**

- The numpydoc format was chosen in part to be aligned with pandas and broader pydata community.
  To that end, the [pandas docstring guide](https://pandas.pydata.org/docs/development/contributing_docstring.html),
  [pandas API reference](https://pandas.pydata.org/docs/reference/index.html),
  and [pandas codebase](https://github.com/pandas-dev/pandas) are excellent stylistic resources.

[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html

### Release notes

Notes about user-facing modifications for all releases (including future releases)
are kept in [RELEASE.md](/RELEASE.md).

Release notes should be updated in pull-requests to help ensure the repository's
history is as transactional as possible.

Contributors should keep release notes concise and always follow existing conventions
and writing style.

**See also**

- [keepachangelog format](https://keepachangelog.com)

### Building the documentation
 TBD