import json
import logging
import subprocess
from functools import total_ordering
from pathlib import Path
from typing import Any, Optional, Union

from flowboost.openfoam.types import FOAMType


class Dictionary:
    """OpenFOAM dictionary file abstraction enabling trivial, Pythonic read
    and write access. A `Dictionary` represents one OpenFOAM dictionary file,
    and all its contents.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Args:
            path (str): Path to the dictionary file, relative to case root: ex. "constant/chemistryProperties".
        """
        self.path: str | Path = path

        # Keywords (-keywords) produces Entries: cache them here
        # Each keyword is a shallow, top-level Entry.
        # The Dictionary, thereby, has a high-level overview of the entire
        # file.
        self._keywords: Optional[list[Entry]] = None

        # Track file changes to dynamically update dictionary contents?
        # Additionally for reverts and clones? Backups could be kept as
        # .-leading filenames (and for diff-based stuff...)
        self._hash: Optional[str] = None
        self._backup: Optional[Dictionary] = None

        # Logging verbosity (push errors to stdout)
        self._verbose: bool = False

    @staticmethod
    def reader(absolute_dictionary_path: str | Path) -> "DictionaryReader":
        """Create a lazy dictionary reader that can be used to programmatically
        access an OpenFOAM dictionary file in a Pythonic manner.

        Args:
            path (str | Path): An absolute path to an OpenFOAM dictionary file, such as
            `"path/to/my/case/0/U"`, or `"my/case/constant/physicalProperties"`.

        Returns:
            DictionaryReader: A dictionary reader operating on a specified file.
        """
        return DictionaryReader(absolute_dictionary_path)

    @staticmethod
    def link(relative_dictionary_path: str) -> "DictionaryLink":
        """Generates a relative, portable link to an OpenFOAM dictionary.

        The link is meant to serve as a floating reference, which enables
        trivial, programmatic modification of the same dictionary field
        across many different case directories.

        The link can be converted to a dictionary reader operating on a
        specific case directory by simply calling
        `DictionaryLink.reader(case_path)`.

        Args:
            path (str): Relative path of an OpenFOAM dictionary file, such as
            `"0/U"` or `"constant/physicalProperties"`.

        Returns:
            DictionaryLink: A portable dictionary linker
        """
        return DictionaryLink(relative_dictionary_path)


class DictionaryReader(Dictionary):
    """A lazy OpenFOAM dictionary reader that can be used to programmatically
    access dictionaries in a Pythonic manner. Uses the `foamDictionary` CLI
    utility, with a very similar API. For additional documentation, refer to
    `dictionary/readme.md`.
    """

    def __init__(self, path):
        super().__init__(path)
        self._keywords: Optional[list[Entry]] = None

    def entry(self, entry: str) -> Optional["Entry"]:
        """Read a specific entry from the Dictionary file. For deep access,
        separate entries with `/` character.

        Args:
            entry (str): Entry to read: ex. `"addLayersControls"`, `"tabulation/tolerance"`.
        """
        if not self._keywords:
            self._discover_keywords()
            if not self._keywords:
                return None

        path_parts = entry.split("/", 1)
        for keyword in self._keywords:
            if keyword.key == path_parts[0]:
                # Found the top-level entry, now navigate deeper if necessary
                if len(path_parts) == 1:
                    return keyword
                else:
                    return keyword.entry(path_parts[1])

        # Entry does not exist
        return None

    def set(self, entry: Union[str, "Entry"], new_value: Any) -> bool:
        """Set the value for an existing dictionary entry.

        Args:
            entry (str): Entry to modify as a /-separated string
            new_value (Any): New value for entry

        Returns:
            bool: True on success, False on failure
        """
        if isinstance(entry, str):
            read = self.entry(entry)
            if not read:
                raise ValueError(f"No such entry in {self}: {entry}")
            entry = read

        if not entry or not isinstance(entry, "Entry"):
            raise ValueError(f"No such entry in {self}: {entry}")

        return entry.set(new_value=new_value)

    def write(self, entry: str, new_value: Any) -> bool:
        """Alias for DictionaryReader.set()

        Args:
            entry (str): Entry to modify as a /-separated string
            new_value (Any): New value for entry

        Returns:
            bool: True on success, False on failure
        """
        return self.set(entry=entry, new_value=new_value)

    def add(self, entry_path: str, value: Any) -> Optional["Entry"]:
        """Adds a new entry to the dictionary at the specified path with the given value."""
        # Split the entry path to find or create the necessary Entry objects
        path_parts = entry_path.split("/")
        current_parent = None
        current_path = []

        for part in path_parts[:-1]:  # Exclude the last part for now
            current_path.append(part)
            found_entry = self.entry("/".join(current_path))
            if found_entry is None:  # If the entry does not exist, create a placeholder
                found_entry = Entry(self, key=part, parent=current_parent)

                if current_parent is None:
                    if self._keywords is None:
                        self._keywords = []

                    self._keywords.append(found_entry)
                else:
                    if current_parent.keywords is None:
                        current_parent.keywords = []

                    current_parent.keywords.append(found_entry)
                current_parent = found_entry
            current_parent = found_entry

        # Now handle the actual entry to add
        new_entry_key = path_parts[-1]
        new_entry = Entry(self, key=new_entry_key, parent=current_parent)

        if current_parent:
            print(f"Adding key={new_entry_key}, parent={current_parent.print_path()}")
        else:
            print(f"Adding key={new_entry_key}, parent=None")

        if current_parent and current_parent.keywords:
            current_parent.keywords.append(new_entry)
        else:
            if self._keywords is None:
                self._keywords = []

            self._keywords.append(new_entry)

        # Execute the CLI command to add the entry with the specified value
        foam_value = FOAMType.to_FOAM(value)
        cmd = ["foamDictionary", self.path, "-entry", entry_path, "-add", foam_value]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.stderr:
            logging.error(f"Error adding new entry '{entry_path}': {result.stderr}")
            return None

        new_entry._value = value
        new_entry._raw_value = foam_value
        print(new_entry)

        return new_entry

    def delete(self, entry_path: str) -> bool:
        """Deletes an entry from the dictionary by its path."""
        target_entry = self.entry(entry_path)
        if target_entry:
            return target_entry.delete()
        else:
            logging.error(f"Cannot delete non-existing entry '{entry_path}'.")
            return False

    def diff(self, against: "Dictionary") -> str:
        """Calculate the diff of two FOAM Dictionary files using
        `foamDictionary -diff`.

        Args:
            to (Dictionary): _description_
        """
        raise NotImplementedError("Diffing not implemented")

    def preload(self):
        """Preloads all keywrods and Entries within the dictionary: expensive,
        used mainly for testing.
        """
        print(f"Discovering keywords for {Path(self.path).name}")
        self._discover_keywords()

        if not self._keywords:
            return

        for kw in self._keywords:
            kw.preload()

    def pretty_print(self):
        if self._keywords:
            for entry in self._keywords:
                print(entry.key)

    def _discover_keywords(self):
        """Discovers top-level keywords in the dictionary."""
        logging.debug(f"Discovering top-level keywords in Dictionary at {self.path}")
        if self._keywords is None:
            self._keywords = []

        cmd = ["foamDictionary", self.path, "-keywords"]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.stderr:
            logging.error(f"Error discovering top-level keywords: {result.stderr}")
            return

        # Process and store the discovered top-level keywords
        for keyword in result.stdout.splitlines():
            entry = Entry(dictionary=self)
            entry.key = keyword
            self._keywords.append(entry)


class DictionaryLink:
    """
    Provides a relative, portable link to an OpenFOAM dictionary. The provided
    path is expected to be a relative path of a dictionary within some a
    hypothetical case directory, such as "constant/chemistryProperties" or
    "0/U".

    Links are meant to serve as a floating references, which enable trivial,
    programmatic modification of the same dictionary entries across many
    different case directories.

    The link can be converted to a dictionary reader operating on a
    specific case directory by simply calling DictionaryLink.reader(case_path).
    """

    def __init__(self, path: str):
        self.path = path
        self.entry_path = ""

    def entry(self, entry_path: str) -> "DictionaryLink":
        """
        Append an entry path to the DictionaryLink, enabling chained calls for deep entry access.
        """
        new_link = DictionaryLink(self.path)
        # If there's already an entry path, append the new one with a separator
        if self.entry_path:
            new_link.entry_path = f"{self.entry_path}/{entry_path}"
        else:
            new_link.entry_path = entry_path
        return new_link

    def index(self, index: int) -> "DictionaryLink":
        """
        Append an index access to the entry path. This is a no-op for the link itself but
        indicates how the reader should process the entry path upon resolution.
        """
        # Simply append the index operation to the entry path; interpretation is deferred
        new_link = DictionaryLink(self.path)
        new_link.entry_path = f"{self.entry_path}[{index}]"
        return new_link

    def reader(
        self, case_path: str | Path
    ) -> Optional[Union[DictionaryReader, "Entry"]]:
        """
        Convert this link into a DictionaryReader that resolves the linked entry path
        within the context of a given case directory.
        """
        full_path = f"{case_path}/{self.path}"
        reader = DictionaryReader(full_path)

        if self.entry_path:
            # If there's an entry path, resolve it to an Entry object
            return reader.entry(self.entry_path)

        return reader

    def __str__(self):
        """
        String representation of the DictionaryLink, mainly for debugging purposes.
        """
        return f"DictionaryLink({self.path}, {self.entry_path})"


@total_ordering
class Entry:
    """OpenFOAM dictionary entry abstraction, serving as a wrapper between
    raw dictionary key-value access.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        key: Optional[str] = None,
        parent: Optional["Entry"] = None,
    ) -> None:
        self.key: Optional[str] = key

        # Read name for dimensioned entries (e.g. nu)
        self._name: Optional[str] = None

        # Plain Pythonic value without dimension (e.g. 1e-5)
        self._value: Optional[Any] = None

        # Dimension: e.g. [1 0 0 ...], [CAD]
        self._dimension: Optional[str] = None

        # Raw value, as read from the dictionary (e.g. "nu [0 2 -1 ...] 1e-5")
        self._raw_value: Optional[Any] = None

        # Track parent and dict file for lazy reads
        self.parent: Optional[Entry] = parent
        self.dictionary: Dictionary = dictionary

        # Is this a terminating entry, or does it have sub-entries?
        self.keywords: Optional[list] = None
        self.terminating: Optional[bool] = None

        self._verbose: bool = self.dictionary._verbose

    @property
    def name(self) -> Optional[str]:
        """Lazy-loads and returns name for dimensioned entries."""
        if self.raw_value is None and self.terminating:
            self._discover_value()

        return self._name

    @property
    def dimension(self) -> Optional[str]:
        if self.raw_value is None and self.terminating:
            self._discover_value()

        # Dimension is set during value discovery
        return self._dimension

    @property
    def value(self):
        """Lazy-loads and returns the post-processed, Pythonic value of the entry."""
        if self._value is None and self.terminating:
            self._discover_value()

        return self._value

    @property
    def raw_value(self):
        """Returns the raw string value of the entry."""
        if self._raw_value is None and self.terminating:
            self._discover_value()

        return self._raw_value

    def index(self, i: int) -> Optional[Any]:
        """Access the n-th index of a vector quantity. Especially useful in the
        case of DictionaryLinks.

        Raises an IndexError if the index does not exist.
        """
        if self.value is None:
            raise ValueError("Entry that does not store a value cannot be indexed")

        return self.value[i]

    def print_path(self) -> str:
        """Constructs the relative path of this dictionary entry."""
        if self.parent:
            return f"{self.parent.print_path()}/{self.key}"
        return self.key or ""

    def entry(self, entry_path: str) -> Optional["Entry"]:
        """Navigates to and returns the requested sub-entry, discovering as needed."""
        # Discover sub-entries if not already done
        if self.keywords is None:
            self.discover_subentries()  # Method to discover and populate self.keywords

        path_parts = entry_path.split("/", 1)
        for sub_entry in self.keywords or []:
            if sub_entry.key == path_parts[0]:
                if len(path_parts) == 1:
                    return sub_entry
                else:
                    return sub_entry.entry(path_parts[1])

        return None

    def set(
        self, new_value: Any, override: bool = False, write_dimensioned: bool = False
    ) -> bool:
        """
        Sets the value of this dictionary entry. For non-terminating entries,
        new_value must be a dictionary, or override must be True.

        Note, that for dimensioned entries `new_value` is expected to only
        represent the value, _without_ the name or dimension. If you would like
        to write an entire dimensioned entry at once (e.g. `"nu [0 2 -1 ...] 1e-5"`
        instead of `1e-5`), set the `write_dimensioned` to `True`.

        Args:
            new_value (Any): The new value to set for the entry.
            override (bool, optional): Whether to override a non-terminating entry. Defaults to False.

        Returns:
            bool: True on success, False on failure.
        """
        if self.terminating is False and not override:
            # For non-terminating entries without override, enforce dictionary type for new_value
            if not isinstance(new_value, dict):
                logging.error(
                    f"Non-terminating entry '{self.print_path()}' requires a dictionary value or override flag."
                )
                return False

            # Additional logic for setting dictionary values: TODO

        if write_dimensioned and not isinstance(new_value, str):
            raise ValueError(
                f"New value must be a string when write_dimensioned=True: {new_value}"
            )

        if write_dimensioned and isinstance(new_value, str):
            # New value is an entire dimensioned entry: infer name, value, dimension
            new_raw_val = new_value
            self._name, new_value, self._dimension = FOAMType.parse(new_raw_val)
        else:
            # Only a value is being written
            foam_value = FOAMType.to_FOAM(new_value)

            # If the entry was dimensioned, re-add name and dimension
            new_raw_val = (
                f"{self._name+' ' if self._name else ''}"
                + f"{self._dimension+' ' if self._dimension else ''}"
                + foam_value
            )

        # For terminating entries or override
        cmd = [
            "foamDictionary",
            self.dictionary.path,
            "-entry",
            self.print_path(),
            "-set",
            new_raw_val,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0 or result.stderr:
            logging.error(
                f"Error setting value for '{self.print_path()}': {result.stderr}"
            )
            return False

        # Assuming success, update the local cached value
        self._raw_value = new_raw_val  # Store the new raw value as string
        self._value = new_value  # Keep Pythonic value

        return True

    def write(self, new_value: Any, override: bool = False) -> bool:
        """Alias for Entry.set()

        Args:
            new_value (Any): _description_

        Returns:
            bool: _description_
        """
        return self.set(new_value=new_value, override=override)

    def add(self, entry_path: str, value: Any) -> bool:
        """Adds a new sub-entry relative to this entry with the given value."""
        rel_path = self.print_path()

        if rel_path:
            full_path = self.print_path() + "/" + entry_path
        else:
            full_path = entry_path

        # return self.dictionary.add(full_path, value)
        return self.add(full_path, value)

    def delete(self) -> bool:
        """Deletes this entry from the dictionary."""
        if self.parent:
            # Remove this entry from the parent's keywords list
            if self.parent.keywords and self in self.parent.keywords:
                self.parent.keywords.remove(self)

        # Execute the CLI command to remove the entry from the dictionary file
        cmd = [
            "foamDictionary",
            self.dictionary.path,
            "-entry",
            self.print_path(),
            "-remove",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.stderr:
            logging.error(
                f"Error deleting entry '{self.print_path()}': {result.stderr}"
            )
            return False

        self._value = None
        self._raw_value = None
        self.keywords = None

        return True

    def pretty_print(self, deep: bool = True, to_stdout: bool = False) -> str | None:
        """Pretty-prints the value of the Entry. Optionally perform a deep
        print, which includes all sub-entries associated with this entry.

        Args:
            deep (bool, optional): Include all sub-entries. Defaults to True.
            to_stdout (bool, optional): Instead of returning a string, call print() directly. Defaults to False.

        Returns:
            str: Formatted and indented string if to_stdout was not set to True.
        """
        if to_stdout:
            print(json.dumps(self.value, indent=2))
        else:
            return json.dumps(self.value, indent=2)

    def preload(self):
        """An expensive, recursive preload of all sub-Entries."""
        print(self)
        if not self.terminating and not self.keywords:
            self.discover_subentries()

    def discover_subentries(self):
        """Discovers sub-entries for this entry, if not already known."""
        if self.terminating is not False:
            # This entry is either known to be terminating or not yet evaluated
            cmd = [
                "foamDictionary",
                self.dictionary.path,
                "-keywords",
                "-entry",
                self.print_path(),
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.stderr:
                self.terminating = True
                return

            if result.stdout.strip():
                # Sub-entries exist
                self.keywords = [
                    Entry(dictionary=self.dictionary, key=line, parent=self)
                    for line in result.stdout.splitlines()
                ]
                self.terminating = False
            else:
                # No sub-entries; this is a terminating entry
                self.terminating = True

    def _add_keyword(self, keyword: str):
        pass

    def _discover_value(self):
        """Retrieves the value of a terminating entry using the foamDictionary CLI and stores both raw and processed values."""
        cmd = [
            "foamDictionary",
            self.dictionary.path,
            "-entry",
            self.print_path(),
            "-value",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.stderr:
            self._value, self._raw_value = None, None
            if self._verbose:
                logging.error(
                    f"Error retrieving value for '{self.print_path()}': {result.stderr}"
                )
            return

        self._raw_value = result.stdout.strip()

        # Do post-processing to Pythonic types with FOAM_Types module
        self._name, self._dimension, self._value = FOAMType.parse(data=self._raw_value)

    def __str__(self):
        # Triggers lazy loading if necessary
        value = self.value
        return f"{self.print_path()}: {value}"

    def __eq__(self, other):
        """Compare this entry to another entry or a value."""
        if isinstance(other, Entry):  # Comparing two Entry objects
            return (self.key == other.key) and (self.parent == other.parent)
        else:  # Comparing the value of this Entry to another value
            return self.value == other

    def __lt__(self, other):
        """Check if the entry's value is less than another value."""
        if isinstance(other, Entry):
            raise NotImplementedError(
                "Less than comparison is not implemented between Entry objects."
            )
        return self.value < other
