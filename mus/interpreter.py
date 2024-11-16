import ast
import operator
import typing as t
from dataclasses import field, make_dataclass
import inspect

class DSLError(Exception):
    def __init__(self, message, node, code):
        self.message = message
        self.line = node.lineno if hasattr(node, 'lineno') else None
        self.col = node.col_offset if hasattr(node, 'col_offset') else None
        self.code = code
        super().__init__(self.format_message())

    def format_message(self):
        location = f"line {self.line}" if self.line is not None else "unknown location"
        if self.col is not None:
            location += f", column {self.col}"
        return f"{self.message} at {location}: {self.code[self.line - 1].strip()}"

class DSLInterpreter:
    def __init__(self, functions: t.Dict[str, t.Callable], variables: t.Optional[t.Dict[str, t.Any]]=None):
        self.variables = {
            **(variables or {}),
            "list": list,
            "dict": dict,
            "int": int,
            "str": str,
            "tuple": tuple,
            "List": t.List,
            "Tuple": t.Tuple,
            "Dict": t.Dict,
            "Union": t.Union,
            "Literal": t.Literal,
            "TypedDict": t.TypedDict,
            "Annotated": t.Annotated,
        }
        self.functions = {
            "range": range,
            **functions
        }
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Mod: operator.mod
        }
        self.code = None

    def parse_and_execute(self, code: str):
        tree = ast.parse(code)
        for node in tree.body:
            result = self.execute_node(node)
            if isinstance(result, ReturnValue):
                return result.value
        return None
        
    def evaluate_type(self, annotation: ast.AST):
        if isinstance(annotation, ast.Name):
            # Handle simple types like int, str, float, etc.
            return self.variables.get(annotation.id)
        elif isinstance(annotation, ast.Constant):
            # Handle simple types like 1, "hello", 3.14, etc.
            return annotation.value
        elif isinstance(annotation, ast.Tuple):
            # Handle tuple types like (int, str), (int, int, int), etc.
            return tuple(self.evaluate_type(elt) for elt in annotation.elts)
        elif isinstance(annotation, ast.Subscript):
            # Handle generic types like List[int], Dict[str, int], etc.
            value = self.evaluate_type(annotation.value)
            slice_value = self.evaluate_type(annotation.slice)
            if value == t.List or value == list:
                return t.List[slice_value]
            elif value == t.Tuple or value == tuple:
                return t.Tuple[slice_value]
            elif value == t.Dict:
                if isinstance(slice_value, ast.Tuple):
                    key_type = self.evaluate_type(slice_value.elts[0])
                    value_type = self.evaluate_type(slice_value.elts[1])
                    return t.Dict[key_type, value_type]
            elif value == t.Union:
                if isinstance(slice_value, ast.Tuple):
                    types = [self.evaluate_type(elt) for elt in slice_value.elts]
                    return t.Union[tuple(types)]
            elif value == t.Literal:
                return t.Literal[slice_value]
            elif value == t.Annotated:
                return t.Annotated[slice_value]
            
        
        raise ValueError(f"Unsupported type annotation: {annotation}")

    def execute_node(self, node: ast.AST):
        try:
            if isinstance(node, ast.FunctionDef):
                self.define_function(node)
            elif isinstance(node, ast.ClassDef):
                self.define_class(node)
            elif isinstance(node, ast.Assign):
                self.assign_variable(node)
            elif isinstance(node, ast.Expr):
                return self.evaluate_expression(node.value)
            elif isinstance(node, ast.For):
                return self.execute_for(node)
            elif isinstance(node, ast.If):
                return self.execute_if(node)
            elif isinstance(node, ast.Return):
                return ReturnValue(self.evaluate_expression(node.value) if node.value else None)
            elif isinstance(node, ast.Pass):
                return None
            elif isinstance(node, ast.Import):
                raise DSLError(f"Syntax error: Import statements are not allowed", node, self.code)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "mus.stubs":
                    return None
                raise DSLError(f"Syntax error: Import statements are not allowed", node, self.code)
            else:
                raise DSLError(f"Syntax error: Unknown statement", node, self.code)
        except DSLError:
            raise
        

    def define_function(self, node: ast.FunctionDef):
        def function(*args, **kwargs):
            local_vars = dict(zip([arg.arg for arg in node.args.args], args))
            old_vars = self.variables.copy()
            self.variables.update(local_vars)
            # FIX: is below correct, given how non-kwargs are handled?
            self.variables.update(kwargs)
            result = None
            for stmt in node.body:
                result = self.execute_node(stmt)
                if isinstance(result, ReturnValue):
                    break
            self.variables = old_vars
            return result.value if isinstance(result, ReturnValue) else result
        
        function.__name__ = node.name

        if not node.body or not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[0].value, ast.Constant):
            raise DSLError(f"Function \"{node.name}\" must have a docstring", node, self.code)
        else:
            function.__doc__ = node.body[0].value.value
        
        annotated_args = {}
        missing_annotations = []
        for arg in node.args.args:
            if arg.annotation is None:
                missing_annotations.append(arg.arg)
            else:
                annotated_args[arg.arg] = self.evaluate_type(arg.annotation)
        if missing_annotations:
            raise DSLError(f"Function \"{node.name}\" arguments must have type annotations for args: {', '.join(missing_annotations)}", node, self.code)
        
        function.__annotations__ = {
            **annotated_args,
            **{"returns": self.evaluate_type(node.returns) if node.returns else None}
        }


        if node.decorator_list:
            for decorator in node.decorator_list[::-1]:
                wrapper = self.evaluate_expression(decorator)
                if not isinstance(wrapper, t.Callable):
                    raise DSLError(f"Decorator must be a callable", decorator, self.code)
                function = wrapper(function)
                    

        
        self.functions[node.name] = function

    def define_class(self, node: ast.ClassDef):
        class_fields = []

        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                field_name = item.target.id
                field_type = self.evaluate_type(item.annotation)
                default_value = self.evaluate_expression(item.value) if item.value else field(default=None)
                class_fields.append((field_name, field_type, field(default=default_value)))
        
        cls = make_dataclass(node.name, class_fields)
        self.variables[node.name] = cls

    def evaluate_expression(self, node: ast.AST):
        try:
            if isinstance(node, (ast.Constant)):
                return node.value
            elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
                return [self.evaluate_expression(elt) for elt in node.elts]
            elif isinstance(node, ast.Dict):
                return {
                    (self.evaluate_expression(key) if key else None): self.evaluate_expression(value)
                    for key, value in zip(node.keys, node.values)
                }
            elif isinstance(node, ast.Name):
                if node.id in self.variables:
                    return self.variables[node.id]
                elif node.id in self.functions:
                    return self.functions[node.id]
                else:
                    raise DSLError(f"Name '{node.id}' is not defined", node, self.code)
            elif isinstance(node, ast.Call):
                func = self.evaluate_expression(node.func)
                args = [self.evaluate_expression(arg) for arg in node.args]
                kwargs = {kw.arg: self.evaluate_expression(kw.value) for kw in node.keywords}
                return func(*args, **kwargs)
            elif isinstance(node, ast.Attribute):
                value = self.evaluate_expression(node.value)
                return getattr(value, node.attr)
            elif isinstance(node, ast.JoinedStr):
                parts = []
                for value in node.values:
                    if isinstance(value, ast.Str):
                        parts.append(value.s)
                    elif isinstance(value, ast.FormattedValue):
                        parts.append(str(self.evaluate_expression(value.value)))
                return ''.join(parts)
            elif isinstance(node, ast.Compare):
                left = self.evaluate_expression(node.left)
                for op, right in zip(node.ops, node.comparators):
                    right_val = self.evaluate_expression(right)
                    if not self.operators[type(op)](left, right_val):
                        return False
                    left = right_val
                return True
            elif isinstance(node, ast.BinOp):
                left = self.evaluate_expression(node.left)
                right = self.evaluate_expression(node.right)
                return self.operators[type(node.op)](left, right)
            elif isinstance(node, ast.Subscript):
                value = self.evaluate_expression(node.value)
                index = self.evaluate_expression(node.slice)
                return value[index]
            elif isinstance(node, ast.Slice):
                lower = self.evaluate_expression(node.lower) if node.lower else None
                upper = self.evaluate_expression(node.upper) if node.upper else None
                step = self.evaluate_expression(node.step) if node.step else None
                return slice(lower, upper, step)
            elif isinstance(node, ast.UnaryOp):
                operand = self.evaluate_expression(node.operand)
                if isinstance(node.op, ast.Not):
                    return operator.not_(operand)
                elif isinstance(node.op, ast.USub):
                    return operator.neg(operand)
                elif isinstance(node.op, ast.Invert):
                    return operator.invert(operand)
                elif isinstance(node.op, ast.UAdd):
                    return operator.pos(operand)
                else:
                    raise SyntaxError(f"Unsupported unary operator: {type(node.op)}")
            
            elif isinstance(node, ast.BoolOp):
                left = self.evaluate_expression(node.values[0])
                right = self.evaluate_expression(node.values[1])
                if isinstance(node.op, ast.And):
                    return left and right
                elif isinstance(node.op, ast.Or):
                    return left or right
                else:
                    raise SyntaxError(f"Unsupported boolean operator: {type(node.op)}")
            else:
                raise SyntaxError(f"Unsupported expression type: {type(node)}")
        except DSLError as e:
            raise e from e
        except Exception as e:
            raise DSLError(str(e), node, self.code) from e


    def assign_variable(self, node: ast.Assign):
        value = self.evaluate_expression(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables[target.id] = value
            elif isinstance(target, ast.Attribute):
                obj = self.evaluate_expression(target.value)
                setattr(obj, target.attr, value)
            else:
                raise SyntaxError(f"Unsupported assignment target: {type(target)}")

    def execute_for(self, node: ast.For):
        iterable = self.evaluate_expression(node.iter)
        if not isinstance(iterable, t.Iterable):
            raise DSLError(f"Object is not iterable", node.iter, self.code)
        for item in iterable:
            self.variables[node.target.id] = item
            for stmt in node.body:
                result = self.execute_node(stmt)
                if isinstance(result, ReturnValue):
                    return result
        return None

    def execute_if(self, node: ast.If):
        if self.evaluate_expression(node.test):
            for stmt in node.body:
                result = self.execute_node(stmt)
                if isinstance(result, ReturnValue):
                    return result
        elif node.orelse:
            for stmt in node.orelse:
                result = self.execute_node(stmt)
                if isinstance(result, ReturnValue):
                    return result
        return None

    def run(self, code: str, variables: t.Optional[t.Dict[str, t.Any]]=None):
        self.code = code.split('\n')
        if variables:
            self.variables.update(variables)
        return self.parse_and_execute(code)

class ReturnValue:
    def __init__(self, value):
        self.value = value