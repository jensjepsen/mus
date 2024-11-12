"""
Fixes types in scripts
"""
import typing as t
if t.TYPE_CHECKING:
    from bob.llm.llm import IterableResult
    from typing import Annotated, Literal, Pattern, Optional, Union, Tuple
    from bob import Bob, File, Query
    from bob.state import State, StateType
    a_bob = Bob()
    state = a_bob.state
    llm = a_bob.llm
    tool = a_bob.tool
    
    __all__ = [
        # Functions
        "state", "llm", "tool",
        # Internal types
        "IterableResult",
        "File",
        "State",
        "Query",
        # Python types
        "Annotated", "Literal", "Pattern", "Optional", "Union", "Tuple"]
else:
    __all__ = []
