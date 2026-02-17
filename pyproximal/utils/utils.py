__all__ = ["Report"]

from types import ModuleType

try:
    # scooby is a soft dependency for pyproximal
    from scooby import Report as ScoobyReport
except ImportError:

    class ScoobyReport:  # type: ignore[no-redef]
        def __init__(
            self,
            additional: list[str | ModuleType] | None,
            core: list[str | ModuleType] | None,
            optional: list[str | ModuleType] | None,
            ncol: int,
            text_width: int,
            sort: bool,
        ) -> None:
            print(
                "\nNOTE: `pyproximal.Report` requires `scooby`. Install it via"
                "\n      `pip install scooby` or "
                "`conda install -c conda-forge scooby`.\n"
            )


class Report(ScoobyReport):  # type: ignore[misc]
    r"""Print date, time, and version information.

    Use ``scooby`` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``,
    ``pylops``, ``pyproximal``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``, ``numba``,
    and ``matplotlib``. It also shows MKL information, if available.

    All modules provided in ``add_pckg`` are also shown.

    .. note::

        The package ``scooby`` has to be installed in order to use ``Report``:
        ``pip install scooby`` or ``conda install -c conda-forge scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from pyproximal import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(
        self,
        add_pckg: list[str | ModuleType] | None = None,
        ncol: int = 3,
        text_width: int = 80,
        sort: bool = False,
    ) -> None:
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core: list[str | ModuleType] | None = ["numpy", "scipy", "pylops", "pyproximal"]

        # Optional packages.
        optional: list[str | ModuleType] | None = ["IPython", "matplotlib", "numba"]

        super().__init__(
            additional=add_pckg,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )
