//! Recursive-descent math expression parser.
//!
//! Ported from `util/string/expression_parser.{h,cpp}`. Produces an AST of
//! [`Node`] variants (`Number`/`Variable`/`Operator`/`Function`) from a string
//! like `"sin(x) + 2*y"`. The AST is consumed by the voxel graph compiler to
//! lower expression strings into runtime graph nodes — there is no evaluator
//! here, only parsing and constant-folding.
//!
//! The C++ implementation uses inheritance (`NumberNode : Node`, etc.); the
//! Rust port collapses all node kinds into a single [`Node`] enum with
//! `Box<Node>` children, which is more idiomatic and avoids `Box<dyn>` virtual
//! dispatch.

/// Operator kinds. Matches `OperatorNode::Operation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

/// AST node. One enum covers every node kind the parser produces. The C++
/// version uses a `Node` base + four subclasses; the Rust port folds that into
/// a single enum with `Box<Node>` children (idiomatic, no vtable cost).
#[derive(Debug, Clone)]
pub enum Node {
    /// Numeric literal.
    Number(f32),
    /// Named variable reference (e.g. `x`).
    Variable(String),
    /// Binary operator. `n0 op n1`.
    Operator {
        op: Op,
        n0: Box<Node>,
        n1: Box<Node>,
    },
    /// Function call. `function_id` references the caller-supplied
    /// [`Function`] table. The parser caps arguments at 4 (matching the C++
    /// `FixedArray<UniquePtr<Node>, 4>`); callers wanting more should extend
    /// the parser.
    Function {
        function_id: u32,
        args: Vec<Box<Node>>,
    },
}

/// Built-in error categories. Matches `ExpressionParser::ErrorID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorId {
    #[default]
    None,
    Invalid,
    UnexpectedEnd,
    InvalidNumber,
    InvalidToken,
    UnexpectedToken,
    UnknownFunction,
    ExpectedArgument,
    TooFewArguments,
    TooManyArguments,
    UnclosedParenthesis,
    MissingOperandArguments,
    MultipleOperands,
}

/// A parse error with context. Matches `ExpressionParser::Error`.
#[derive(Debug, Clone, Default)]
pub struct Error {
    pub id: ErrorId,
    /// The byte offset in the source string where the error was detected.
    pub position: usize,
    /// The offending symbol (function name, unexpected token text, etc.).
    pub symbol: String,
}

impl Error {
    fn new(id: ErrorId, position: usize) -> Self {
        Self {
            id,
            position,
            symbol: String::new(),
        }
    }

    fn with_symbol(id: ErrorId, position: usize, symbol: impl Into<String>) -> Self {
        Self {
            id,
            position,
            symbol: symbol.into(),
        }
    }
}

/// Result of [`parse`]. On success `root` is `Some`; on error `error.id != None`.
#[derive(Debug, Default)]
pub struct ParseResult {
    pub root: Option<Box<Node>>,
    pub error: Error,
}

/// Caller-supplied function callback used by the constant-folder. Matches
/// `ExpressionParser::FunctionCallback`.
pub type FunctionCallback = fn(&[f32]) -> f32;

/// A function the parser recognises. The caller registers a table of these
/// (one per graph node that can appear in an expression); the parser matches
/// by `name`, validates argument count, and stores the resolved `id` in the
/// [`Node::Function`]. The `func` pointer is only used by the constant-folder.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    pub name: &'static str,
    pub argument_count: u32,
    pub id: u32,
    pub func: Option<FunctionCallback>,
}

/// Parses an expression string into an AST. The AST is constant-folded in
/// place: any subtree whose operands are all `Number` literals collapses to a
/// single `Number`. Returns a [`ParseResult`] carrying either `root` or `error`.
///
/// Matches `ExpressionParser::parse`.
pub fn parse(text: &str, functions: &[Function]) -> ParseResult {
    let mut tokenizer = Tokenizer::new(text);
    let mut result = parse_expression(&mut tokenizer, false, functions, None);
    if result.error.id != ErrorId::None {
        return result;
    }
    if let Some(root) = result.root.take() {
        result.root = Some(precompute_constants(root, functions));
    }
    result
}

/// Returns a list of unique variable names referenced in `node`, in
/// first-occurrence order. Matches `ExpressionParser::find_variables`.
pub fn find_variables(node: &Node, variables: &mut Vec<String>) {
    match node {
        Node::Number(_) => {}
        Node::Variable(name) => {
            if !variables.iter().any(|v| v == name) {
                variables.push(name.clone());
            }
        }
        Node::Operator { n0, n1, .. } => {
            find_variables(n0, variables);
            find_variables(n1, variables);
        }
        Node::Function { args, .. } => {
            for arg in args {
                find_variables(arg, variables);
            }
        }
    }
}

/// Pretty-prints the AST for debugging. Matches `ExpressionParser::tree_to_string`.
pub fn tree_to_string(node: &Node, functions: &[Function]) -> String {
    let mut out = String::new();
    write_node(node, functions, &mut out);
    out
}

fn write_node(node: &Node, functions: &[Function], out: &mut String) {
    match node {
        Node::Number(v) => {
            use std::fmt::Write;
            let _ = write!(out, "{v}");
        }
        Node::Variable(name) => out.push_str(name),
        Node::Operator { op, n0, n1 } => {
            out.push('(');
            write_node(n0, functions, out);
            out.push(' ');
            out.push_str(op_str(*op));
            out.push(' ');
            write_node(n1, functions, out);
            out.push(')');
        }
        Node::Function { function_id, args } => {
            let name = functions
                .iter()
                .find(|f| f.id == *function_id)
                .map(|f| f.name)
                .unwrap_or("<unknown>");
            out.push_str(name);
            out.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_node(arg, functions, out);
            }
            out.push(')');
        }
    }
}

fn op_str(op: Op) -> &'static str {
    match op {
        Op::Add => "+",
        Op::Subtract => "-",
        Op::Multiply => "*",
        Op::Divide => "/",
        Op::Power => "^",
    }
}

/// Returns `true` if the two trees are structurally equal (same shape, same
/// operators, same numbers, same variable names, same function ids). Matches
/// `ExpressionParser::is_tree_equal`.
pub fn is_tree_equal(a: &Node, b: &Node) -> bool {
    match (a, b) {
        (Node::Number(x), Node::Number(y)) => x.to_bits() == y.to_bits(),
        (Node::Variable(x), Node::Variable(y)) => x == y,
        (
            Node::Operator {
                op: op_a,
                n0: a0,
                n1: a1,
            },
            Node::Operator {
                op: op_b,
                n0: b0,
                n1: b1,
            },
        ) => op_a == op_b && is_tree_equal(a0, b0) && is_tree_equal(a1, b1),
        (
            Node::Function {
                function_id: id_a,
                args: aa,
            },
            Node::Function {
                function_id: id_b,
                args: ba,
            },
        ) => {
            id_a == id_b
                && aa.len() == ba.len()
                && aa.iter().zip(ba).all(|(x, y)| is_tree_equal(x, y))
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f32),
    Name(String),
    Plus,
    Minus,
    Divide,
    Multiply,
    ParenthesisOpen,
    ParenthesisClose,
    Power,
    Comma,
}

struct Tokenizer<'a> {
    chars: &'a [u8],
    position: usize,
    error: ErrorId,
}

impl<'a> Tokenizer<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            chars: text.as_bytes(),
            position: 0,
            error: ErrorId::None,
        }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn error(&self) -> ErrorId {
        self.error
    }

    fn get_next(&mut self) -> Option<Token> {
        while self.position < self.chars.len() {
            let c = self.chars[self.position];
            match c {
                b' ' | b'\t' => {
                    self.position += 1;
                }
                b'(' => {
                    self.position += 1;
                    return Some(Token::ParenthesisOpen);
                }
                b')' => {
                    self.position += 1;
                    return Some(Token::ParenthesisClose);
                }
                b',' => {
                    self.position += 1;
                    return Some(Token::Comma);
                }
                b'+' => {
                    self.position += 1;
                    return Some(Token::Plus);
                }
                b'-' => {
                    self.position += 1;
                    return Some(Token::Minus);
                }
                b'*' => {
                    self.position += 1;
                    return Some(Token::Multiply);
                }
                b'/' => {
                    self.position += 1;
                    return Some(Token::Divide);
                }
                b'^' => {
                    self.position += 1;
                    return Some(Token::Power);
                }
                _ if is_name_starter(c) => {
                    let name = self.take_name();
                    return Some(Token::Name(name));
                }
                _ if is_digit(c) => match self.take_number() {
                    Ok(value) => return Some(Token::Number(value)),
                    Err(()) => {
                        self.error = ErrorId::InvalidNumber;
                        return None;
                    }
                },
                _ => {
                    self.error = ErrorId::InvalidToken;
                    return None;
                }
            }
        }
        None
    }

    fn take_name(&mut self) -> String {
        let start = self.position;
        while self.position < self.chars.len() && is_name_char(self.chars[self.position]) {
            self.position += 1;
        }
        // Names are ASCII-only (the C++ parser matches `[A-Za-z_]` then
        // `[A-Za-z0-9_]`); using `from_utf8` here is safe in practice.
        String::from_utf8(self.chars[start..self.position].to_vec()).unwrap_or_default()
    }

    fn take_number(&mut self) -> Result<f32, ()> {
        let start = self.position;
        while self.position < self.chars.len() && is_digit(self.chars[self.position]) {
            self.position += 1;
        }
        let int_end = self.position;
        let mut float_end = int_end;
        if float_end < self.chars.len() && self.chars[float_end] == b'.' {
            float_end += 1;
            self.position = float_end;
            while self.position < self.chars.len() && is_digit(self.chars[self.position]) {
                self.position += 1;
            }
            float_end = self.position;
        }
        let bytes = &self.chars[start..float_end];
        let text = std::str::from_utf8(bytes).map_err(|_| ())?;
        // Parse manually so the C++ behaviour (integer + decimal parts) is
        // preserved exactly: an `int_end == start` slice is an error, while
        // the standard parser would reject things like `12.` differently.
        let int_part: f32 = std::str::from_utf8(&self.chars[start..int_end])
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .ok_or(())?;
        let mut value = int_part;
        if float_end != int_end {
            // Decimal part after the `.`.
            let decimal_text =
                std::str::from_utf8(&self.chars[int_end + 1..float_end]).map_err(|_| ())?;
            if !decimal_text.is_empty() {
                let decimals: f32 = decimal_text.parse::<f32>().map_err(|_| ())?;
                let scale = 10f32.powi(decimal_text.len() as i32);
                value += decimals / scale;
            }
        }
        // Reject identifiers glued to numbers (`12abc`).
        if self.position < self.chars.len() && is_name_starter(self.chars[self.position]) {
            return Err(());
        }
        let _ = text;
        Ok(value)
    }
}

fn is_name_starter(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn is_name_char(c: u8) -> bool {
    is_name_starter(c) || is_digit(c)
}

// ---------------------------------------------------------------------------
// Parser (recursive descent with operator-precedence stack)
// ---------------------------------------------------------------------------

const MAX_PRECEDENCE: i32 = 100;

fn op_precedence(op: Op) -> i32 {
    match op {
        Op::Add | Op::Subtract => 1,
        Op::Multiply | Op::Divide => 2,
        Op::Power => 3,
    }
}

fn token_to_op(token: &Token) -> Option<Op> {
    match token {
        Token::Plus => Some(Op::Add),
        Token::Minus => Some(Op::Subtract),
        Token::Multiply => Some(Op::Multiply),
        Token::Divide => Some(Op::Divide),
        Token::Power => Some(Op::Power),
        _ => None,
    }
}

struct OpEntry {
    precedence: i32,
    op: Op,
}

#[allow(clippy::vec_box)]
fn pop_expression_operator(
    operations_stack: &mut Vec<OpEntry>,
    operand_stack: &mut Vec<Box<Node>>,
) -> Result<(), ErrorId> {
    let last_op = operations_stack.pop().expect("operations stack non-empty");
    if operand_stack.len() < 2 {
        return Err(ErrorId::MissingOperandArguments);
    }
    // All current operators are binary: right popped first, then left, so the
    // operand order on the stack (left below right) is preserved.
    let right = operand_stack.pop().unwrap();
    let left = operand_stack.pop().unwrap();
    operand_stack.push(Box::new(Node::Operator {
        op: last_op.op,
        n0: left,
        n1: right,
    }));
    Ok(())
}

#[allow(clippy::manual_pop_if)]
fn parse_expression(
    tokenizer: &mut Tokenizer,
    in_argument_list: bool,
    functions: &[Function],
    out_last_token: Option<&mut Token>,
) -> ParseResult {
    let mut operations_stack: Vec<OpEntry> = Vec::new();
    let mut operand_stack: Vec<Box<Node>> = Vec::new();
    let mut precedence_base = 0i32;
    let mut previous_was_operand = false;
    let mut last_token: Option<Token> = None;

    while let Some(t) = tokenizer.get_next() {
        if in_argument_list && matches!(t, Token::Comma) {
            last_token = Some(t);
            break;
        }

        let mut current_is_operand = false;

        if let Some(op) = token_to_op(&t) {
            let precedence = precedence_base + op_precedence(op);
            while let Some(last) = operations_stack.last() {
                if precedence <= last.precedence {
                    if let Err(id) =
                        pop_expression_operator(&mut operations_stack, &mut operand_stack)
                    {
                        return ParseResult {
                            root: None,
                            error: Error::new(id, tokenizer.position()),
                        };
                    }
                } else {
                    break;
                }
            }
            operations_stack.push(OpEntry { precedence, op });
        } else if matches!(t, Token::Number(_) | Token::Name(_)) {
            if previous_was_operand {
                return ParseResult {
                    root: None,
                    error: Error::new(ErrorId::MultipleOperands, tokenizer.position()),
                };
            }
            operand_stack.push(operand_to_node(t));
            current_is_operand = true;
        } else if matches!(t, Token::ParenthesisOpen) {
            let mut is_function_call = false;
            if let Some(back) = operand_stack.last() {
                if matches!(back.as_ref(), Node::Variable(_)) {
                    let fn_name =
                        if let Node::Variable(name) = operand_stack.pop().unwrap().as_ref() {
                            name.clone()
                        } else {
                            unreachable!()
                        };
                    match parse_function(tokenizer, fn_name, functions) {
                        Ok(node) => operand_stack.push(Box::new(node)),
                        Err(error) => {
                            return ParseResult { root: None, error };
                        }
                    }
                    is_function_call = true;
                }
            }
            if !is_function_call {
                precedence_base += MAX_PRECEDENCE;
            }
        } else if matches!(t, Token::ParenthesisClose) {
            last_token = Some(t.clone());
            if in_argument_list && precedence_base < MAX_PRECEDENCE {
                break;
            }
            if precedence_base < MAX_PRECEDENCE {
                return ParseResult {
                    root: None,
                    error: Error::new(ErrorId::UnexpectedToken, tokenizer.position()),
                };
            }
            precedence_base -= MAX_PRECEDENCE;
        } else {
            return ParseResult {
                root: None,
                error: Error::new(ErrorId::UnexpectedToken, tokenizer.position()),
            };
        }

        previous_was_operand = current_is_operand;
    }

    if let Err(id) = tokenizer.error().try_into_none() {
        return ParseResult {
            root: None,
            error: Error::new(id, tokenizer.position()),
        };
    }

    if precedence_base != 0 {
        return ParseResult {
            root: None,
            error: Error::new(ErrorId::UnclosedParenthesis, tokenizer.position()),
        };
    }

    if let Some(slot) = out_last_token {
        if let Some(t) = last_token {
            *slot = t;
        }
    }

    while !operations_stack.is_empty() {
        if let Err(id) = pop_expression_operator(&mut operations_stack, &mut operand_stack) {
            return ParseResult {
                root: None,
                error: Error::new(id, tokenizer.position()),
            };
        }
    }

    debug_assert!(operand_stack.len() <= 1);
    ParseResult {
        root: operand_stack.pop(),
        error: Error::default(),
    }
}

fn operand_to_node(token: Token) -> Box<Node> {
    match token {
        Token::Number(v) => Box::new(Node::Number(v)),
        Token::Name(name) => Box::new(Node::Variable(name)),
        _ => unreachable!("operand_to_node only handles Number/Name"),
    }
}

fn parse_function(
    tokenizer: &mut Tokenizer,
    name: String,
    functions: &[Function],
) -> std::result::Result<Node, Error> {
    let function = functions.iter().find(|f| f.name == name);
    let Some(function) = function else {
        return Err(Error::with_symbol(
            ErrorId::UnknownFunction,
            tokenizer.position(),
            name,
        ));
    };

    let mut args: Vec<Box<Node>> = Vec::with_capacity(function.argument_count as usize);
    let mut last_token = Token::ParenthesisOpen;

    for arg_index in 0..function.argument_count {
        let arg_result = parse_expression(tokenizer, true, functions, Some(&mut last_token));
        if arg_result.error.id != ErrorId::None {
            return Err(arg_result.error);
        }
        let Some(root) = arg_result.root else {
            return Err(Error::with_symbol(
                ErrorId::ExpectedArgument,
                tokenizer.position(),
                &name,
            ));
        };
        if matches!(last_token, Token::ParenthesisClose) && arg_index + 1 < function.argument_count
        {
            return Err(Error::with_symbol(
                ErrorId::TooFewArguments,
                tokenizer.position(),
                &name,
            ));
        }
        args.push(root);
    }

    if !matches!(last_token, Token::ParenthesisClose) {
        return Err(Error::with_symbol(
            ErrorId::TooManyArguments,
            tokenizer.position(),
            &name,
        ));
    }

    Ok(Node::Function {
        function_id: function.id,
        args,
    })
}

/// Recursively constant-folds an AST. Returns the (possibly rewritten) node
/// and, when the entire subtree folds to a literal, the computed value via
/// `out_number`. The function is structured to take and return ownership so
/// the borrow checker stays happy across recursive calls.
///
/// Returns `(node, Some(value))` when the node is fully constant; in that
/// case `node` is replaced with `Node::Number(value)`. Returns `(node, None)`
/// when the node contains variables.
fn precompute_constants(node: Box<Node>, functions: &[Function]) -> Box<Node> {
    match *node {
        Node::Number(_) | Node::Variable(_) => node,
        Node::Operator { op, n0, n1 } => {
            let n0_folded = precompute_constants(n0, functions);
            let n1_folded = precompute_constants(n1, functions);
            match (n0_folded.as_ref(), n1_folded.as_ref()) {
                (Node::Number(a), Node::Number(b)) => {
                    let v = match op {
                        Op::Add => a + b,
                        Op::Subtract => a - b,
                        Op::Multiply => a * b,
                        Op::Divide => a / b,
                        Op::Power => a.powf(*b),
                    };
                    Box::new(Node::Number(v))
                }
                _ => Box::new(Node::Operator {
                    op,
                    n0: n0_folded,
                    n1: n1_folded,
                }),
            }
        }
        Node::Function { function_id, args } => {
            let folded_args: Vec<Box<Node>> = args
                .into_iter()
                .map(|a| precompute_constants(a, functions))
                .collect();
            let all_constant = folded_args
                .iter()
                .all(|a| matches!(a.as_ref(), Node::Number(_)));
            if all_constant {
                if let Some(f) = functions.iter().find(|f| f.id == function_id) {
                    if let Some(callback) = f.func {
                        let constants: Vec<f32> = folded_args
                            .iter()
                            .map(|a| {
                                let Node::Number(v) = a.as_ref() else {
                                    unreachable!()
                                };
                                *v
                            })
                            .collect();
                        return Box::new(Node::Number(callback(&constants)));
                    }
                }
            }
            Box::new(Node::Function {
                function_id,
                args: folded_args,
            })
        }
    }
}

// Small helper trait to convert ErrorId into a Result-friendly type.
trait ErrorIdExt {
    fn try_into_none(self) -> std::result::Result<(), ErrorId>;
}
impl ErrorIdExt for ErrorId {
    fn try_into_none(self) -> std::result::Result<(), ErrorId> {
        if self == ErrorId::None {
            Ok(())
        } else {
            Err(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sin_fn(args: &[f32]) -> f32 {
        args[0].sin()
    }

    fn sample_functions() -> Vec<Function> {
        vec![Function {
            name: "sin",
            argument_count: 1,
            id: 1,
            func: Some(sin_fn),
        }]
    }

    #[test]
    fn parses_simple_arithmetic_with_variable_to_preserve_tree_shape() {
        // Use a variable so the constant-folder doesn't collapse the tree;
        // pure-literal expressions get folded (see constant_folds_*).
        let result = parse("x + 2 * 3", &[]);
        let root = result.root.unwrap();
        // Precedence: x + (2*3). The 2*3 sub-expression IS folded to 6.
        match *root {
            Node::Operator {
                op: Op::Add,
                n0,
                n1,
            } => {
                assert!(matches!(*n0, Node::Variable(_)));
                assert!(matches!(*n1, Node::Number(6.0)));
            }
            other => panic!("expected add operator, got {other:?}"),
        }
    }

    #[test]
    fn constant_folds_when_all_operands_are_literals() {
        let result = parse("1 + 2 * 3", &[]);
        assert_eq!(result.error.id, ErrorId::None);
        // 1 + 6 = 7, folded into a single Number node.
        match *result.root.unwrap() {
            Node::Number(v) => assert_eq!(v, 7.0),
            other => panic!("expected folded number, got {other:?}"),
        }
    }

    #[test]
    fn keeps_variable_subtrees_intact() {
        let result = parse("x * 2 + 1", &[]);
        match *result.root.unwrap() {
            Node::Operator { op: Op::Add, .. } => {}
            other => panic!("expected add at the root, got {other:?}"),
        }
    }

    #[test]
    fn parenthesised_subexpression_overrides_precedence() {
        let result = parse("(1 + 2) * 3", &[]);
        // 3 * 3 = 9, fully folded.
        match *result.root.unwrap() {
            Node::Number(v) => assert_eq!(v, 9.0),
            other => panic!("expected folded number, got {other:?}"),
        }
    }

    #[test]
    fn parses_function_call_with_constant_argument() {
        let result = parse("sin(0)", &sample_functions());
        // sin(0) = 0, folded via the registered callback.
        match *result.root.unwrap() {
            Node::Number(v) => assert!((v - 0.0).abs() < 1e-6),
            other => panic!("expected folded sin(0)=0, got {other:?}"),
        }
    }

    #[test]
    fn parses_function_call_with_variable_argument() {
        let result = parse("sin(x)", &sample_functions());
        match *result.root.unwrap() {
            Node::Function { function_id, args } => {
                assert_eq!(function_id, 1);
                assert_eq!(args.len(), 1);
                assert!(matches!(*args[0], Node::Variable(_)));
            }
            other => panic!("expected function node, got {other:?}"),
        }
    }

    #[test]
    fn errors_on_unknown_function() {
        let result = parse("foo(x)", &[]);
        assert_eq!(result.error.id, ErrorId::UnknownFunction);
        assert_eq!(result.error.symbol, "foo");
    }

    #[test]
    fn errors_on_too_few_arguments() {
        let funcs = vec![Function {
            name: "add",
            argument_count: 2,
            id: 7,
            func: None,
        }];
        let result = parse("add(1)", &funcs);
        assert_eq!(result.error.id, ErrorId::TooFewArguments);
    }

    #[test]
    fn errors_on_unclosed_parenthesis() {
        let result = parse("(1 + 2", &[]);
        assert_eq!(result.error.id, ErrorId::UnclosedParenthesis);
    }

    #[test]
    fn errors_on_invalid_token() {
        let result = parse("1 + @", &[]);
        assert_eq!(result.error.id, ErrorId::InvalidToken);
    }

    #[test]
    fn find_variables_returns_unique_names_in_order() {
        let result = parse("a + b * a + sin(c)", &sample_functions());
        let root = result.root.unwrap();
        let mut vars = Vec::new();
        find_variables(&root, &mut vars);
        assert_eq!(
            vars,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn tree_to_string_round_trips_simple_expression() {
        let result = parse("a + b", &[]);
        let root = result.root.unwrap();
        let printed = tree_to_string(&root, &[]);
        assert!(printed.contains("a"));
        assert!(printed.contains("b"));
        assert!(printed.contains('+'));
    }

    #[test]
    fn is_tree_equal_compares_structure() {
        let a = parse("x + y", &[]).root.unwrap();
        let b = parse("x + y", &[]).root.unwrap();
        let c = parse("x * y", &[]).root.unwrap();
        assert!(is_tree_equal(&a, &b));
        assert!(!is_tree_equal(&a, &c));
    }

    #[test]
    fn decimal_numbers_parse_correctly() {
        let result = parse("1.5 + 0.5", &[]);
        // 1.5 + 0.5 = 2.0, folded.
        match *result.root.unwrap() {
            Node::Number(v) => assert!((v - 2.0).abs() < 1e-6),
            other => panic!("expected folded number, got {other:?}"),
        }
    }
}
