//! Expression node for the voxel graph.
//!
//! Ports `NODE_EXPRESSION` from C++ `VoxelGraphFunction`. An expression node
//! takes a string like `"0.1 * x + 0.2 * z + min(y, 0.5)"` and evaluates it
//! using the [`expression_parser`](crate::string::expression_parser).
//!
//! The node maps named variables (e.g., `"x"`, `"y"`, `"z"`) to graph input
//! ports. At evaluation time, each slice element's input values are bound to
//! the variable names, and the parsed expression is evaluated.

use crate::string::expression_parser::{self, Function, Node as AstNode};

/// An expression node that evaluates a string expression at each voxel.
///
/// The expression is parsed once at construction, then evaluated per-voxel.
/// Variable names map to input port indices.
#[derive(Debug, Clone)]
pub struct ExpressionNode {
    /// The parsed AST root (None if parse failed).
    ast: Option<Box<AstNode>>,
    /// Variable name → input port index mapping.
    /// E.g., `[("x", 0), ("y", 1), ("z", 2)]` means port 0 → "x", etc.
    variable_ports: Vec<(String, usize)>,
    /// The original expression string.
    expression_text: String,
}

impl ExpressionNode {
    /// Create a new expression node from a text expression.
    /// `variables` maps variable names to input port indices.
    pub fn new(expression: &str, variables: &[(&str, usize)]) -> Result<Self, String> {
        let functions = [
            Function {
                name: "min",
                argument_count: 2,
                id: 0,
                func: Some(|args: &[f32]| args[0].min(args[1])),
            },
            Function {
                name: "max",
                argument_count: 2,
                id: 1,
                func: Some(|args: &[f32]| args[0].max(args[1])),
            },
            Function {
                name: "abs",
                argument_count: 1,
                id: 2,
                func: Some(|args: &[f32]| args[0].abs()),
            },
            Function {
                name: "sin",
                argument_count: 1,
                id: 3,
                func: Some(|args: &[f32]| args[0].sin()),
            },
            Function {
                name: "cos",
                argument_count: 1,
                id: 4,
                func: Some(|args: &[f32]| args[0].cos()),
            },
            Function {
                name: "sqrt",
                argument_count: 1,
                id: 5,
                func: Some(|args: &[f32]| args[0].sqrt()),
            },
        ];

        let result = expression_parser::parse(expression, &functions);
        if result.error.id != expression_parser::ErrorId::None {
            return Err(format!("expression parse error: {:?}", result.error));
        }

        let var_ports: Vec<(String, usize)> = variables
            .iter()
            .map(|(name, port)| (name.to_string(), *port))
            .collect();

        Ok(Self {
            ast: result.root,
            variable_ports: var_ports,
            expression_text: expression.to_string(),
        })
    }

    /// Evaluate the expression for a single set of input values.
    /// `inputs` is indexed by port index.
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        let ast = match &self.ast {
            Some(a) => a,
            None => return 0.0,
        };
        eval_node(ast.as_ref(), &self.variable_ports, inputs)
    }

    /// Evaluate the expression for a slice of input values (one per element).
    /// Returns a Vec<f32> of results.
    pub fn evaluate_slice(&self, input_ports: &[&[f32]]) -> Vec<f32> {
        let n = input_ports.first().map(|s| s.len()).unwrap_or(0);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let inputs: Vec<f32> = (0..self.variable_ports.len())
                .map(|p| {
                    let port = self.variable_ports[p].1;
                    input_ports
                        .get(port)
                        .and_then(|s| s.get(i))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            result.push(self.evaluate(&inputs));
        }
        result
    }

    /// The original expression string.
    pub fn expression_text(&self) -> &str {
        &self.expression_text
    }

    /// Whether the expression parsed successfully.
    pub fn is_valid(&self) -> bool {
        self.ast.is_some()
    }
}

fn eval_node(node: &AstNode, vars: &[(String, usize)], inputs: &[f32]) -> f32 {
    use expression_parser::{Node as AstNode, Op};
    match node {
        AstNode::Number(n) => *n,
        AstNode::Variable(name) => {
            for (var_name, port) in vars {
                if var_name == name {
                    return inputs.get(*port).copied().unwrap_or(0.0);
                }
            }
            0.0
        }
        AstNode::Operator { op, n0, n1 } => {
            let l = eval_node(n0.as_ref(), vars, inputs);
            let r = eval_node(n1.as_ref(), vars, inputs);
            match op {
                Op::Add => l + r,
                Op::Subtract => l - r,
                Op::Multiply => l * r,
                Op::Divide => {
                    if r.abs() < 1e-30 {
                        0.0
                    } else {
                        l / r
                    }
                }
                Op::Power => l.powf(r),
            }
        }
        AstNode::Function { function_id, args } => {
            let evaluated: Vec<f32> = args
                .iter()
                .map(|a| eval_node(a.as_ref(), vars, inputs))
                .collect();
            match *function_id {
                0 => evaluated[0].min(evaluated[1]),
                1 => evaluated[0].max(evaluated[1]),
                2 => evaluated[0].abs(),
                3 => evaluated[0].sin(),
                4 => evaluated[0].cos(),
                5 => evaluated[0].sqrt(),
                _ => 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_arithmetic() {
        let node = ExpressionNode::new("1 + 2", &[]).unwrap();
        assert!((node.evaluate(&[]) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn variable_substitution() {
        let node = ExpressionNode::new("x * 2", &[("x", 0)]).unwrap();
        assert!((node.evaluate(&[5.0]) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn complex_expression() {
        // 0.1 * x + 0.2 * z + min(y, 0.5)
        let node = ExpressionNode::new(
            "0.1 * x + 0.2 * z + min(y, 0.5)",
            &[("x", 0), ("y", 1), ("z", 2)],
        )
        .unwrap();
        let result = node.evaluate(&[10.0, 1.0, 5.0]);
        // 0.1*10 + 0.2*5 + min(1.0, 0.5) = 1.0 + 1.0 + 0.5 = 2.5
        assert!((result - 2.5).abs() < 1e-5, "complex expression: {result}");
    }

    #[test]
    fn min_max_functions() {
        let node = ExpressionNode::new("min(3, 7) + max(2, 5)", &[]).unwrap();
        assert!((node.evaluate(&[]) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn abs_function() {
        let node = ExpressionNode::new("abs(0 - 5)", &[]).unwrap();
        assert!((node.evaluate(&[]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn slice_evaluation() {
        let node = ExpressionNode::new("x + 1", &[("x", 0)]).unwrap();
        let xs = [0.0f32, 1.0, 2.0, 3.0];
        let result = node.evaluate_slice(&[&xs]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn invalid_expression_returns_error() {
        let result = ExpressionNode::new("x +", &[("x", 0)]);
        assert!(result.is_err());
    }

    #[test]
    fn divide_small_denominator_finite() {
        let node = ExpressionNode::new("1 / 0.001", &[]).unwrap();
        let v = node.evaluate(&[]);
        assert!(v.is_finite() && v > 0.0, "should be finite positive: {v}");
    }
}
