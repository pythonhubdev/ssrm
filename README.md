![Build](https://github.com/optimizely/ssrm/workflows/Build/badge.svg)

# Sequential Sample Ratio Mismatch (SRM) test.
A package for sequential testing of Sample Ratio Mismatch (SRM).

Contributors:
- Michael Lindon (michael.lindon@optimizely.com )

## Installation
We recommend that you use an isolated virtual environment to install and run the code in this repo (See: [virtualenv](https://pypi.org/project/virtualenv/) and [pyenv](https://github.com/pyenv/pyenv))

1. Install dependencies: Run `make install`.
    - If you wish to develop in the repo, run `make
    install-dev`.  Also, see the contributing doc [here](https://github.com/optimizely/ssrm/blob/master/CONTRIBUTING.md)
    > **Tip:** have a look in the [`Makefile`](https://github.com/optimizely/ssrm/blob/master/Makefile) to learn more about what this, and other make recipes do!
1. Run tests:
    -   `make check` to run all checks.
    -   `make test` to run unit tests.


## Tutorials
We provide a tutorial notebook that walks through an example of running a
Sequential SRM test
[here](https://github.com/optimizely/ssrm/blob/master/notebooks/introduction.ipynb).  Run `jupyter lab`, and open `notebooks/introduction.ipynb`.

## Documentation
The latest reference documentation is here (TBD).

## Contributing
See the contributing doc [here](https://github.com/optimizely/ssrm/blob/master/CONTRIBUTING.md).
