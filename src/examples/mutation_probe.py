"""Mutation probe example for Tok runtime validation."""


def scale_numbers(numbers: list[int], factor: int) -> list[int]:
    """Multiply each number in the list by factor."""
    return [n * factor for n in numbers]


class MutationProbe:
    """Applies a fixed scale transformation to integer data."""

    def __init__(self, label: str) -> None:
        """Initialize the mutation probe with a label."""
        self.label = label

    def run(self, data: list[int]) -> list[int]:
        """Run the mutation probe on the input data."""
        return scale_numbers(data, factor=3)


if __name__ == "__main__":
    # runtime validation example
    probe = MutationProbe(label="probe-1")
    result = scale_numbers([2, 4, 6], factor=2)
