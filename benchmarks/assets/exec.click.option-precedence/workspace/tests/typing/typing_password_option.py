import codecs

import click
from typing_extensions import assert_type


@click.command()
@click.password_option()
def encrypt(password: str) -> None:
    click.echo(f"encoded: to {codecs.encode(password, 'rot13')}")


assert_type(encrypt, click.Command)
