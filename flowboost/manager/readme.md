# `Manager` job scheduler interface
The Manager module provides a simple, abstract job submission and monitoring interface. The manager class implements a non-abstract persistent job monitoring function (`do_monitoring()`), which leverages the abstract methods such as `_job_has_finished()`. This allows for effectively any cluster scheduler to be used as a backend.

## Cluster scheduler interfaces
The Manager-class interfaces with the cluster scheduler using abstract methods, which are implemented under `manager/interfaces`. These interfaces are not required to do any state tracking, as this is automatically done by the base class.

To implement a new scheduler interface, the following four methods must be implemented according to their definitions under the `Manager` base class:

1. `_is_available() -> bool`
2. `_submit_job() -> Optional[str]`
3. `_cancel_job() -> bool`
4. `_job_has_finished() -> bool`

All methods are internal, and are not meant to be called by the user. However, how these methods are implemented is left up to the user. For examples, see `manager/interfaces/local.py` and `manager/interfaces/sge.py`.
