# OpenFOAM interfaces
FlowBoost comes bundled with a suite of OpenFOAM utilities for easy, Pythonic interfacing to FOAM case data.

## Case abstraction
The Case abstraction serves as a simple interface for modifying and accessing OpenFOAM case data.

### Persistence
The cases use TOML-files for persisting certain key information, which ensures portability between optimization campaigns, while also reducing the level of statefulness required by the optimizer. An example file appears as follows:

```toml
name = "newCaseDir"
id = "762c94f1"
path = "path/to/newCaseDir"
status = "not_submitted"
success = false
created_at = 1979-05-27T07:32:00Z
```

## Function object data access
FlowBoost provides easy access to function object output data using two DataFrame backends: the new and shiny, SQL-like Polars, and Pandas.

## Dictionary
Dictionary, available under `dictionary.Dictionary`, provides easy read/write access to OpenFOAM dictionary files and their entries. The backend uses `foamDictionary`, with a very similar API based around just a few commands:

-  A dictionary file (`Dictionary`)
   -  A reader for interfacing with a specific file: `Dictionary.reader() -> DictionaryReader`
   -  A dictionary link serving as a relative reference, that can be converted to a reader by providing a base path: `Dictionary.link() -> DictionaryLink -> DictionaryLink.reader(case_dir) -> DictionaryReader`

-  Entries within the dictionary file (`Entry`)
   -  Set entry values (`Entry.set()`)
   -  Add entry values (`Dictionary.add()`)
   -  Delete entries (`Entry.delete()`, `Dictionary.delete()`)

The following example serves as a simple introduction:

```python
from dictionary import Dictionary
from foam_case import Case

# Example case
my_case = Case.from_tutorial("multicomponentFluid/aachenBomb", "my_case_dir")

# Reader provides access to entries within the dictionary file
# Read directly from a dictionary path: two equivalent ways
reader = Dictionary.reader(my_case.path / "constant/chemistryProperties")
reader = my_case.dictionary("constant/chemistryProperties")

# Try priting an entry
print(reader.entry("odeCoeffs/solver")) # 'odeCoeffs/solver: seulex'

# Access value
my_val = reader.entry("odeCoeffs/eps").value * 3.14

# Comparisons access the value behind the scenes: keep in mind that the Entry
# is still an object, not a value!
assert reader.entry("odeCoeffs/eps") == 0.05 # OK

# Change the value
reader.entry("odeCoeffs/eps").set(0.1)
reader.entry("odeCoeffs/eps").write(0.05) # Alias for "Entry.set"

# Add a value
reader.add("FoamFile/myCoolKey", 69.420)

# Remove a value, whichever way seems natural
reader.entry("FoamFile/myCoolKey").delete() # Entry.delete
reader.delete("FoamFile/myCoolKey") # Dictionary.delete, same result
```
