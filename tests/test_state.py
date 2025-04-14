import pytest
from mus.state import StateReference, State, Empty, encode_obj
import jsonpickle

@pytest.fixture
def state_manager():
    return State()

def test_state_initialization():
    state = StateReference(42)
    assert state() == 42

def test_state_update():
    state = StateReference(42)
    assert state(100) == 100
    assert state() == 100

def test_state_empty():
    state = StateReference(42)
    assert state(Empty()) == 42

def test_state_to_dict():
    state = StateReference(42)
    assert state.to_dict() == {"val": 42}

def test_state_from_dict():
    data = {"val": 42}
    state = StateReference.from_dict(data)
    assert state() == 42

@pytest.mark.parametrize("value", [
    42,
    "hello",
    [1, 2, 3],
    {"a": 1, "b": 2},
    None
])
def test_state_with_different_types(value):
    state = StateReference(value)
    assert state() == value
    assert state.to_dict() == {"val": value}

def test_encode_obj():
    class DummyObj:
        def to_dict(self):
            return {"dummy": "value"}

    assert encode_obj(DummyObj()) == {"dummy": "value"}
    assert encode_obj(42) == 42

def test_state_manager_init(state_manager):
    state = state_manager.init("test", 42)
    assert state() == 42
    assert "test" in state_manager.states
    assert "test" in state_manager.is_set

def test_state_manager_init_loaded(state_manager):
    state_manager.loads('{"test": {"val": 100}}')
    state1 = state_manager.init("test", 42)
    assert state1() == 100
    
    
def test_state_manager_dumps(state_manager):
    state_manager.init("int_state", 42)
    state_manager.init("str_state", "hello")
    dumped = state_manager.dumps()
    decoded = jsonpickle.decode(dumped)
    assert decoded == {
        "int_state": {"val": 42},
        "str_state": {"val": "hello"}
    }

def test_state_manager_loads(state_manager):
    data = jsonpickle.encode({
        "int_state": {"val": 42},
        "str_state": {"val": "hello"}
    })
    state_manager.loads(data)
    assert state_manager.states["int_state"]() == 42
    assert state_manager.states["str_state"]() == "hello"

def test_state_manager_multiple_states(state_manager):
    state1 = state_manager.init("state1", 42)
    state2 = state_manager.init("state2", "hello")
    state3 = state_manager.init("state3", [1, 2, 3])

    assert state1() == 42
    assert state2() == "hello"
    assert state3() == [1, 2, 3]

    state1(100)
    state2("world")
    state3([4, 5, 6])

    assert state1() == 100
    assert state2() == "world"
    assert state3() == [4, 5, 6]

def test_state_manager_call_to_init(state_manager):
    state1 = state_manager("state1", 42)
    state2 = state_manager("state2", "hello")
    state3 = state_manager("state3", [1, 2, 3])

    assert state1() == 42
    assert state2() == "hello"
    assert state3() == [1, 2, 3]

def test_state_manager_is_set(state_manager):
    state_manager.init("test", 42)
    assert "test" in state_manager.is_set
    state_manager.init("test", 100)
    assert "test" in state_manager.is_set

def test_state_manager_dumps_loads_roundtrip(state_manager):
    state_manager.init("int_state", 42)
    state_manager.init("str_state", "hello")
    state_manager.init("list_state", [1, 2, 3])

    dumped = state_manager.dumps()
    
    new_state_manager = State()
    new_state_manager.loads(dumped)

    assert new_state_manager.states["int_state"]() == 42
    assert new_state_manager.states["str_state"]() == "hello"
    assert new_state_manager.states["list_state"]() == [1, 2, 3]