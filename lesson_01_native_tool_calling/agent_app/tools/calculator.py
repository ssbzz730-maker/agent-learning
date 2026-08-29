"""使用 AST 白名单实现的安全数学计算器。"""

import ast
import operator


MAX_ABSOLUTE_VALUE = 10**15
MAX_EXPRESSION_LENGTH = 100
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _check_value(value):
    """拒绝布尔值、非数字和可能造成资源消耗的超大结果。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("表达式只能包含数字")
    if abs(value) > MAX_ABSOLUTE_VALUE:
        raise ValueError("计算结果过大")
    return value


def _evaluate(node):
    """递归计算通过白名单校验的 AST 节点。"""

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _check_value(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return _check_value(UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("指数不能超过 10")
        return _check_value(BINARY_OPERATORS[type(node.op)](left, right))
    raise ValueError("表达式包含不允许的语法")


def calculate(expression):
    """计算只包含数字和基础运算符的表达式，不使用危险的 eval。"""

    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression 必须是非空字符串")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("表达式过长")
    try:
        tree = ast.parse(expression, mode="eval")
        return _evaluate(tree)
    except (SyntaxError, ZeroDivisionError) as error:
        raise ValueError(f"无效的数学表达式：{error}") from error
