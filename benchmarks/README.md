## SSRM Benchmark

Benchmarking SSRM with Airspeed Velocity.


### Requirements

Install `airspeed velocity` and `virtualenv`

Although `asv` ships with support for both `anaconda` and `virtualenv`, this repo will run `asv` inside a virtual environment using `virtualenv`:

```
    pip install asv
    pip install virtualenv
```

### Usage

After making changes to functions in `ssrm_test.py`, commit your change and run the following commands to record results and generate HTML in a graphical view:

```
    cd benchmarks
    asv run
    asv publish
    asv preview
```

More on how to use ``asv`` can be found in [ASV documentation](https://asv.readthedocs.io/)

Command-line help is available as usual via `asv --help` and
`asv run --help`.
