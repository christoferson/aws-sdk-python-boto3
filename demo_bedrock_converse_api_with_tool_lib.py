import json

import ast
import operator as op

ToolIDefinition = {
    "toolSpec": {
        "name": "expr_evaluator",
        "description": """Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input
math expressions.""",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Numerical Expresion. Example 47.5 + 98.3."
                    }
                },
                "required": [
                    "expression"
                ]
            }
        }
    }
}


# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr):
    """
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value  # integer
    elif isinstance(node, ast.BinOp):
        left = eval_(node.left)
        right = eval_(node.right)
        return operators[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = eval_(node.operand)
        return operators[type(node.op)](operand)
    else:
        raise TypeError(node)

#def eval_(node):
#    match node:
#        case ast.Constant(value) if isinstance(value, int):
#            return value  # integer
#        case ast.BinOp(left, op, right):
#            return operators[type(op)](eval_(left), eval_(right))
#        case ast.UnaryOp(op, operand):  # e.g., -1
#            return operators[type(op)](eval_(operand))
#        case _:
#            raise TypeError(node)
   