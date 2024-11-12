import pytest
from mus.interpreter import DSLInterpreter, DSLError

@pytest.fixture
def interpreter():
    return DSLInterpreter({}, {})

def test_slice(interpreter):
    code = """
numbers = [1, 2, 3, 4, 5]
first_two = numbers[:2]
last_two = numbers[3:]
middle_three = numbers[1:4]
last_three = numbers[-3:]
first_three = numbers[:-2]
    """
    interpreter.run(code)
    assert interpreter.variables['first_two'] == [1, 2]
    assert interpreter.variables['last_two'] == [4, 5]
    assert interpreter.variables['middle_three'] == [2, 3, 4]
    assert interpreter.variables['last_three'] == [3, 4, 5]
    assert interpreter.variables['first_three'] == [1, 2, 3]

def test_basic_arithmetic(interpreter):
    code = """
a = 5 + 3
b = 10 - 2
c = 4 * 3
d = 20 / 5
    """
    interpreter.run(code)
    assert interpreter.variables['a'] == 8
    assert interpreter.variables['b'] == 8
    assert interpreter.variables['c'] == 12
    assert interpreter.variables['d'] == 4

def test_function_definition_and_call(interpreter):
    code = """
def add(x: int, y: int) -> int:
    "This function adds two numbers"
    return x + y

result = add(5, 3)
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == 8
    assert interpreter.functions['add'].__annotations__ == {'x': int, 'y': int, 'returns': 'int'}
    assert interpreter.functions['add'].__doc__ == "This function adds two numbers"

def test_class_definition(interpreter):
    code = """
class Person:
    name: Annotated[str, "The name of the person"]
    age: int = 0

p = Person(name="Alice", age=30)
p2 = Person(name="Bob")
    """
    interpreter.run(code)
    assert interpreter.variables['p'].name == "Alice"
    assert interpreter.variables['p'].age == 30
    assert interpreter.variables['p2'].name == "Bob"
    assert interpreter.variables['p2'].age == 0

def test_for_loop(interpreter):
    code = """
sum = 0
for i in [1, 2, 3, 4, 5]:
    sum = sum + i
    """
    interpreter.run(code)
    assert interpreter.variables['sum'] == 15

def test_if_statement(interpreter):
    code = """
x = 10
if x > 5:
    result = "Greater"
else:
    result = "Less or Equal"
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == "Greater"

def test_complex_types(interpreter):
    code = """
numbers = [1, 2, 3]
person = {"name": "Bob", "age": 25}
    """
    interpreter.run(code)
    assert interpreter.variables['numbers'] == [1, 2, 3]
    assert interpreter.variables['person'] == {"name": "Bob", "age": 25}

def test_error_handling():
    interpreter = DSLInterpreter({}, {})
    with pytest.raises(DSLError) as excinfo:
        interpreter.run("undefined_variable")
    assert "Name 'undefined_variable' is not defined" in str(excinfo.value)

def test_return_value(interpreter):
    code = """
def get_value():
    '''This function returns 42'''
    return 42

result = get_value()
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == 42

def test_nested_function_calls(interpreter):
    code = """
def outer(x: int):
    '''Outer function'''
    def inner(y: int):
        '''Inner function'''
        return y * 2
    return inner(x) + 5

result = outer(3)
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == 11

def test_string_operations(interpreter):
    code = """
greeting = "Hello"
name = "World"
message = greeting + " " + name
    """
    interpreter.run(code)
    assert interpreter.variables['message'] == "Hello World"

def test_comparison_operators(interpreter):
    code = """
a = 5
b = 10
less_than = a < b
greater_than = a > b
equal_to = a == 5
    """
    interpreter.run(code)
    assert interpreter.variables['less_than'] == True
    assert interpreter.variables['greater_than'] == False
    assert interpreter.variables['equal_to'] == True

def test_logical_operators(interpreter):
    code = """
a = True
b = False
and_result = a and b
or_result = a or b
not_result = not a
    """
    interpreter.run(code)
    assert interpreter.variables['and_result'] == False, f"Expected False, got {interpreter.variables['and_result']}"
    assert interpreter.variables['or_result'] == True, f"Expected True, got {interpreter.variables['or_result']}"
    assert interpreter.variables['not_result'] == False, f"Expected False, got {interpreter.variables['not_result']}"

def test_unary_operators(interpreter):
    code = """
a = 5
b = -a
c = +a
    """
    interpreter.run(code)
    assert interpreter.variables['a'] == 5
    assert interpreter.variables['b'] == -5
    assert interpreter.variables['c'] == 5

def test_import_not_allowed(interpreter):
    code = """
import os
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Syntax error" in str(excinfo.value)

    code = """
from os import path
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Syntax error" in str(excinfo.value)

def test_import_of_stubs_allowed(interpreter):
    code = """
from mus.stubs import *
    """
    interpreter.run(code)

def test_eval_not_allowed(interpreter):
    code = """
result = eval("2 + 2")
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Name 'eval' is not defined" in str(excinfo.value)

def test_exec_not_allowed(interpreter):
    code = """
exec("print('Hello')")
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Name 'exec' is not defined" in str(excinfo.value)

def test_file_operations_not_allowed(interpreter):
    code = """
open('test.txt', 'w')
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Name 'open' is not defined" in str(excinfo.value)

# Tests for returning from inside loops and control blocks

def test_return_from_for_loop(interpreter):
    code = """
def find_even():
    '''Find the first even number in a list'''
    for i in [1, 2, 3, 4, 5]:
        if i % 2 == 0:
            return i
    return None

result = find_even()
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == 2

def test_return_from_nested_if(interpreter):
    code = """
def check_number(x: int):
    '''Check if a number is positive, even, odd or non-positive'''
    if x > 0:
        if x % 2 == 0:
            return "Positive even"
        else:
            return "Positive odd"
    else:
        return "Non-positive"

result1 = check_number(4)
result2 = check_number(3)
result3 = check_number(-1)
    """
    interpreter.run(code)
    assert interpreter.variables['result1'] == "Positive even"
    assert interpreter.variables['result2'] == "Positive odd"
    assert interpreter.variables['result3'] == "Non-positive"

def test_functions_must_have_docstring(interpreter):
    code = """
def add(x: int, y: int) -> int:
    return x + y
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Function \"add\" must have a docstring" in str(excinfo.value)

def test_function_args_must_have_annotations(interpreter):
    code = """
def add(x, y) -> int:
    "This function adds two numbers"
    return x + y
    """
    with pytest.raises(DSLError) as excinfo:
        interpreter.run(code)
    assert "Function \"add\" arguments must have type annotations for args: x, y" in str(excinfo.value)

def test_applying_decorators():
    def decorator(func):
        "A simple decorator"
        def wrapper(x: int, y: int) -> int:
            "Wrapper function"
            result = func(x, y*2)
            return result
        return wrapper
    interpreter = DSLInterpreter({"decorator": decorator}, {})
    code = """
@decorator
def add(x: int, y: int) -> int:
    "This function adds two numbers"
    return x + y

result = add(5, 3)
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == 11

    class ClassWithDecorator:
        def decorator(self, func):
            "A simple decorator"
            def wrapper(x: int, y: int) -> int:
                "Wrapper function"
                result = func(x, y*3)
                return result
            return wrapper
    interpreter = DSLInterpreter({"ClassWithDecorator": ClassWithDecorator}, {})

    code = """
@ClassWithDecorator().decorator
def add(x: int, y: int) -> int:
    "This function adds two numbers"
    return x + y

result = add(5, 3)
    """
    interpreter.run(code)
    assert interpreter.variables['result'] == 14
