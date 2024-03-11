import re
from enum import Enum
from typing import Any, Optional, Union

import numpy as np


class FOAMType:
    @staticmethod
    def parse(data: str):
        # Attempt to directly return the parsed scalar types
        scalar_value = FOAMType.try_parse_scalar(data)

        if scalar_value is not None:
            return scalar_value

        # TODO test if data is dimensioned

        # Handle non-scalar types with a separate method
        if data.startswith('(') and data.endswith(')'):
            return FOAMType.parse_vector_space(data)

        # Attempt to parse as boolean
        boolean_value = Switch.from_string(data).value
        if boolean_value is not None:
            return boolean_value

        # If all else fails, return the data as is
        return data

    @staticmethod
    def to_FOAM(data: Any) -> str:
        # Helper function to handle vectors and tensors
        def format_vector_or_tensor(d, shape):
            if shape == (3,):  # Vector
                return f"( {d[0]} {d[1]} {d[2]} )"
            elif shape == (3, 3):  # Tensor
                flattened = d.flatten() if isinstance(d, np.ndarray) else [
                    num for row in d for num in row]
                return "( " + " ".join(str(num) for num in flattened) + " )"

            flattened = np.array(d).flatten() if not isinstance(
                d, np.ndarray) else d.flatten()
            return "( " + " ".join(str(num) for num in flattened) + " )"

        # Direct conversion for simple types
        if isinstance(data, str):
            return data
        if isinstance(data, bool):
            return str(data).lower()
        if isinstance(data, (int, float)):
            return str(data)

        # Distinguish between numpy array and list/tuple
        if isinstance(data, np.ndarray):
            shape = data.shape
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            # Assuming all elements in list/tuple are of the same type and shape
            shape = (len(data), len(data[0])) if isinstance(
                data[0], (list, tuple)) else (len(data),)
        else:
            raise TypeError("Unsupported type for FOAMType conversion.")

        # Handle vectors and tensors
        return format_vector_or_tensor(data, shape)

    @staticmethod
    def try_parse_scalar(s: str) -> Optional[Union[int, float]]:
        """
        A time-efficient implementation of a standard str -> (int | float)
        converter. Almost 3x as fast as a standard try-except int->float->None
        converter:

        ```python
        def try_except_scalar(s: str):
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return None
        ```

        `try_except_scalar`: 1138.8 ns/op.
        `try_parse_scalar`: 428.8 ns/op.

        Args:
            s (str): String to try converting

        Returns:
            optional(int, float): Parsed string
        """
        # Check for empty string early.
        if not s:
            return None

        # Strip leading and trailing whitespaces.
        s = s.strip()

        if not s:
            return None

        # Explicit check for special floating-point values
        if s in {"NaN", "nan", "inf", "-inf", "+inf"}:
            return float(s)

        # Check if the string represents an integer value.
        if s.isdigit() or (s[0] in "+-" and s[1:].isdigit()):
            return int(s)

        if '.' in s or 'e' in s or 'E' in s:
            try:
                # Directly return if float conversion succeeds.
                return float(s)
            except ValueError:
                # If conversion fails, it's not a valid float or scientific notation.
                return None

        # If none of the above, the string does not represent a numeric value.
        return None

    @staticmethod
    def parse_vector_space(data: str, parse_subdicts=True):
        # Clean up and prepare for parsing
        data = data.strip().strip('()').strip()

        # Check for sub-dictionaries
        if '{' in data:
            if parse_subdicts:
                return [FOAMType.parse_subdict(data)]
            else:
                return [data]  # Return as string if not parsing

        numbers = [FOAMType.try_parse_scalar(
            num) for num in re.split(r'\s+', data)]

        if len(numbers) == 1:
            return numbers[0]  # Spherical Tensor
        elif len(numbers) == 3:
            return np.array(numbers)  # Vector
        elif len(numbers) == 6:
            # Symmetrical Tensor
            return FOAMType.construct_symm_tensor(numbers)
        elif len(numbers) == 9:
            return np.array(numbers).reshape((3, 3))  # Tensor
        else:
            return np.array(numbers)  # Fallback, just in case

    @staticmethod
    def parse_subdict(data: str) -> dict:
        # Very rudimentary parser for sub-dictionaries
        # Assumes well-formed input because I'm not writing a full parser here
        subdict_str = data.strip('{}').strip()
        entries = re.split(r';\s*', subdict_str)
        parsed_dict = {}
        for entry in entries:
            if entry:  # Skip empty strings
                key, value = entry.split(maxsplit=1)
                key, value = key.strip(), value.strip()

                # Attempt to parse value as vector if it looks like one
                if value.startswith('(') and value.endswith(')'):
                    parsed_dict[key] = FOAMType.parse_vector_space(
                        value, False)
                else:
                    # Try to convert numerical values, fall back to string
                    if value.isdigit():
                        parsed_dict[key] = float(value)
                    else:
                        parsed_dict[key] = value

        return parsed_dict

    @staticmethod
    def construct_symm_tensor(components):
        assert len(components) == 6, "Symmetrical tensor must have 6 components"
        tensor = np.zeros((3, 3))
        indices = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for (i, j), component in zip(indices, components):
            tensor[i, j] = tensor[j, i] = component
        return tensor


class Switch(Enum):
    FALSE = False
    TRUE = True
    NO = False
    YES = True
    OFF = False
    ON = True
    INVALID = None  # Use None for invalid to explicitly indicate an unhandled case

    @classmethod
    def from_string(cls, value: str):
        return cls.__members__.get(value.upper(), cls.INVALID)
