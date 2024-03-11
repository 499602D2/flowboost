import logging
from typing import Any, Literal, Optional, Type, Union

from flowboost.openfoam.dictionary import DictionaryLink


class Dimension:
    def __init__(self, name: str, type: Literal["range", "fixed", "choice"]):
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
        """ Links a search space dimension to a corresponding OpenFOAM
        dictionary entry, which gets manipulated during the optimization

        Args:
            dictionary_entry (Entry): An OpenFOAM Dictionary entry. The entry
            should be initialized as a `Dictionary().Entry` to ensure the full,
            relative path is available.
        """
        self.linked_entry = dictionary_link
        logging.info(f"Linked dim='{self.name}' to {DictionaryLink}")

    @classmethod
    def range(cls,
              name: str,
              link: DictionaryLink,
              lower: Union[int, float],
              upper: Union[int, float],
              log_scale: bool = False,
              dtype: Type = float,
              digits: Optional[int] = None) -> 'Dimension':
        dim = cls(name, "range")
        dim.linked_entry = link
        dim.bounds = [lower, upper]
        dim.log_scale = log_scale
        dim.value_type = Dimension._get_value_type_str(dtype)
        dim.digits = digits
        return dim

    @classmethod
    def fixed(cls,
              name: str,
              link: DictionaryLink,
              value: Union[int, float, str, bool]) -> 'Dimension':
        dim = cls(name, "fixed")
        dim.linked_entry = link
        dim.values = [value]  # Using list to unify the handling of values
        dim.value_type = Dimension._get_value_type_str(
            Dimension._infer_type(dim.values)
        )
        return dim

    @classmethod
    def choice(cls,
               name: str,
               link: DictionaryLink,
               choices: list[Union[int, float, str, bool]],
               dtype: Optional[Type] = None,
               is_ordered: bool = False) -> 'Dimension':
        dim = cls(name, "choice")
        dim.linked_entry = link
        dim.is_ordered = is_ordered

        # If dtype is provided, ensure all choices match this type, otherwise infer it
        if dtype is None:
            dtype = cls._infer_type(choices)

        # Ensure all choices match the determined dtype
        dim.values = [cls._ensure_types_match(
            choice, dtype) for choice in choices]
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

    @staticmethod
    def _get_value_type_str(value_type: Type) -> str:
        """
        Converts a type (e.g., int, float, bool, str) to its string representation.
        """
        type_map = {
            int: "int",
            float: "float",
            bool: "bool",
            str: "str"
        }

        # Check if the value_type is in the type_map and return its string repr
        if value_type in type_map:
            return type_map[value_type]
        else:
            raise ValueError(
                f"Unsupported type {value_type}, must be (int, float, bool, str)")
