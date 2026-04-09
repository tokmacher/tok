import click
from typing_extensions import assert_type


@click.group(context_settings={})
def hello() -> None:
    pass


assert_type(hello, click.Group)
