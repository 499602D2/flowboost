import logging
import math
from typing import Any, Literal, Optional, Type, Union

from flowboost.openfoam.dictionary import DictionaryLink

# Target significant-digit headroom when auto-picking `digits` for a float
# range. Float64 carries ~15-17 significant digits, so 12 leaves room below
# the last-bit noise that appears when BO converges on a box-constraint
# boundary and would otherwise defeat Ax's hash-based arm deduplication.
_DEFAULT_FLOAT_SIG_DIGITS = 12


class Dimension:
    """A search space dimension linked to an OpenFOAM dictionary entry.
    Create via ``Dimension.range()``, ``Dimension.fixed()``, or ``Dimension.choice()``."""

    def __init__(self, name: str, type: Literal["range", "fixed", "choice"]):
        if " " in name:
            raise ValueError(f"Dimension name cannot contain spaces: '{name}'")
        self.name = name
        self.type = type
        self.value_type: Optional[str] = None
        self.is_fidelity: Optional[bool] = None
        self.target_value: Optional[float] = None
        self.is_ordered: Optional[bool] = None
        self.is_task: Optional[bool] = None
        self.log_scale: Optional[bool] = False
        self.digits: Optional[int] = None
        self.values: Optional[list[Union[int, float, str, bool]]] = None
        self.bounds: Optional[list[Union[int, float]]] = None

        # Link to an OpenFOAM dictionary entry
        self.linked_entry: Optional[DictionaryLink] = None

    def link_to(self, dictionary_link: DictionaryLink):
        """Links a search space dimension to a corresponding OpenFOAM
        dictionary entry, which gets manipulated during the optimization

        Args:
            dictionary_entry (Entry): An OpenFOAM Dictionary entry. The entry
            should be initialized as a `Dictionary().Entry` to ensure the full,
            relative path is available.
        """
        self.linked_entry = dictionary_link
        logging.info(f"Linked dim='{self.name}' to {dictionary_link}")

    @classmethod
    def range(
        cls,
        name: str,
        link: DictionaryLink,
        lower: Union[int, float],
        upper: Union[int, float],
        log_scale: bool = False,
        dtype: Type = float,
        digits: Optional[int] = None,
    ) -> "Dimension":
        """Create a range dimension.

        When ``dtype is float`` and ``digits`` is not supplied, a
        magnitude-aware default is picked so rounded values preserve ~12
        significant digits while stripping the last-bit float noise that
        appears when BO converges on a box-constraint boundary (see
        ``_default_digits_for_bounds``). Pass ``digits`` explicitly to
        override, or ``digits=-1`` to disable rounding entirely.
        """
        dim = cls(name, "range")
        dim.linked_entry = link
        dim.bounds = [lower, upper]
        dim.log_scale = log_scale
        dim.value_type = Dimension._get_value_type_str(dtype)

        if digits is None and dtype is float:
            digits = _default_digits_for_bounds(lower, upper)
        elif digits is not None and digits < 0:
            digits = None

        dim.digits = digits
        return dim

    @classmethod
    def fixed(
        cls, name: str, link: DictionaryLink, value: Union[int, float, str, bool]
    ) -> "Dimension":
        dim = cls(name, "fixed")
        dim.linked_entry = link
        dim.values = [value]  # Using list to unify the handling of values
        dim.value_type = Dimension._get_value_type_str(
            Dimension._infer_type(dim.values)
        )
        return dim

    @classmethod
    def choice(
        cls,
        name: str,
        link: DictionaryLink,
        choices: list[Union[int, float, str, bool]],
        dtype: Optional[Type] = None,
        is_ordered: bool = False,
    ) -> "Dimension":
        dim = cls(name, "choice")
        dim.linked_entry = link
        dim.is_ordered = is_ordered

        # If dtype is provided, ensure all choices match this type, otherwise infer it
        if dtype is None:
            dtype = cls._infer_type(choices)

        # Ensure all choices match the determined dtype
        dim.values = [cls._ensure_types_match(choice, dtype) for choice in choices]
        dim.value_type = Dimension._get_value_type_str(dtype)

        return dim

    @staticmethod
    def _infer_type(values: list[Any]) -> Type:
        """
        Infers the data type from the first element in the list, then checks
        to ensure all elements can be converted to this type.
        """
        if not values:
            raise ValueError("Cannot infer type from an empty list.")

        # Use the type of the first element as the reference type
        inferred_type = type(values[0])

        # Check all elements can be converted to the inferred type
        for value in values:
            if not isinstance(value, inferred_type):
                try:
                    Dimension._ensure_types_match(value, inferred_type)
                except ValueError:
                    raise ValueError(f"Incompatible value ({value}) in list")
        return inferred_type

    @staticmethod
    def _ensure_types_match(value: Any, target_type: Type) -> Any:
        """
        Converts the given value to the target_type, if possible and necessary.
        """
        if isinstance(value, target_type):
            return value

        try:
            if target_type is bool and isinstance(value, str):
                # Convert strings to bool explicitly (assuming 'True', 'False' strings)
                return value.lower() in ["true", "1", "t", "y", "yes"]
            else:
                return target_type(value)
        except ValueError:
            logging.error(f"Cannot convert {value} to {target_type}.")
            raise

    # Canonical mapping between Python types and their string representations.
    # Used by _get_value_type_str (type→str) and coerce (str→type).
    _TYPE_MAP: dict[type, str] = {int: "int", float: "float", bool: "bool", str: "str"}
    _TYPE_MAP_INV: dict[str, type] = {v: k for k, v in _TYPE_MAP.items()}

    @staticmethod
    def _get_value_type_str(value_type: Type) -> str:
        """
        Converts a type (e.g., int, float, bool, str) to its string representation.
        """
        if value_type in Dimension._TYPE_MAP:
            return Dimension._TYPE_MAP[value_type]

        raise ValueError(
            f"Unsupported type {value_type}, must be (int, float, bool, str)"
        )

    def coerce_value(self, value: Any) -> Any:
        """Coerce *value* to this dimension's declared ``value_type``.

        For float-typed dimensions with ``digits`` set, the result is also
        rounded to that many decimal places. This mirrors what Ax's
        ``RangeParameter`` does on generated values and, crucially, also
        applies to values coming back from OpenFOAM dictionaries or being
        re-attached via ``attach_trial`` — which Ax does *not* round. Without
        this, BO-near-boundary float noise slips past ``Arm.md5hash`` dedup.

        Returns *value* unchanged when ``value_type`` is ``None``.
        """
        if self.value_type is None:
            return value
        coerced = Dimension._coerce(value, self.value_type)
        if self.value_type == "float" and self.digits is not None:
            coerced = round(coerced, self.digits)
        return coerced

    @staticmethod
    def _coerce(value: Any, value_type: str) -> Any:
        """Coerce *value* to the Python type indicated by *value_type*.

        Handles the quirks that arise when values are read back from
        OpenFOAM dictionaries or TOML metadata (strings, numpy scalars,
        ``bool`` being a subclass of ``int``, etc.).

        Args:
            value: The raw value to coerce.
            value_type: One of ``"int"``, ``"float"``, ``"bool"``, ``"str"``.

        Returns:
            The value converted to the corresponding Python native type.

        Raises:
            ValueError: If *value_type* is unknown or conversion fails.
        """
        target = Dimension._TYPE_MAP_INV.get(value_type)
        if target is None:
            raise ValueError(f"Unknown value_type: {value_type!r}")

        # Exact type match — skip conversion.
        # NOTE: ``isinstance`` is intentionally avoided because
        # ``isinstance(True, int)`` is True in Python.
        if type(value) is target:
            return value

        if target is int:
            # int(float(v)) handles string integers ("3") and float→int
            coerced = int(float(value))
            if float(value) != coerced:
                logging.warning(f"Lossy int coercion: {value!r} truncated to {coerced}")
            return coerced

        if target is bool:
            return Dimension._ensure_types_match(value, bool)

        return target(value)


def _default_digits_for_bounds(
    lower: Union[int, float], upper: Union[int, float]
) -> int:
    """Pick a decimal-place count that gives ``_DEFAULT_FLOAT_SIG_DIGITS``
    significant digits across *lower* and *upper*.

    Ax's ``digits`` is a count of decimal places, so a fixed default cuts
    both ways — fine for ``[500, 2000]``, catastrophic for ``[1e-9, 1e-7]``.
    This scales with the bounds' magnitude so small-valued dimensions keep
    their precision.
    """
    max_magnitude = max(abs(float(lower)), abs(float(upper)))
    if max_magnitude == 0:
        return _DEFAULT_FLOAT_SIG_DIGITS
    order = math.floor(math.log10(max_magnitude))
    return max(0, _DEFAULT_FLOAT_SIG_DIGITS - (order + 1))
