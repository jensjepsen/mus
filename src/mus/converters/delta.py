import cattrs
import cattrs.preconf.json
from ..llm.types import DeltaText, DeltaHistory, DeltaToolResult, DeltaToolUse, Delta, ToolUse, ToolResult
import json

delta_converter = cattrs.preconf.json.make_converter()

if __name__ == "__main__":
    test = DeltaHistory(
        data=[
            Delta(content=DeltaText(data="Hello")),
            Delta(content=DeltaToolUse(data=ToolUse(name="test", input={"a": 1}, id="1"))),
            Delta(content=DeltaToolResult(data=ToolResult(id="1", content="World"))),
        ]
    )
    delta_dict = delta_converter.unstructure(test)
    print(json.dumps(delta_dict, indent=2))

    restored = delta_converter.structure(delta_dict, DeltaHistory)
    print(restored)