import pathlib
import typing

__version__: typing.Final[typing.Text] = (
    pathlib.Path(__file__).parent.parent.parent.joinpath("VERSION").read_text().strip()
)
