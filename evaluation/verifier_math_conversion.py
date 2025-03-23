from sympy import parse_expr, Eq, latex, Symbol, Number
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, TokenError, standard_transformations, implicit_multiplication_application
import re

def is_number(element: str) -> bool:
    try:
        float(element.replace(" ", ""))
        return True
    except ValueError:
        return False
    
def percentage_to_fraction(text):
    pattern = r"(\d+(\.\d+)?%)"
    matches = re.findall(pattern, text)
    for match in matches:
        percentage_str = match[0]
        percentage = float(percentage_str.strip("%")) / 100
        fraction = str(percentage)
        text = text.replace(percentage_str, fraction)
    return text


def remove_outer_parentheses(expr_str):
    expr_str = expr_str.strip()  # Remove leading/trailing whitespace first
    # remove outer parentheses
    expr_str = re.sub(r'^\\\[\s*(.+?)\s*\\\]$', r'\1', expr_str)  # Remove \[ \]
    # expr_str = re.sub(r'^\[\s*(.+?)\s*\]$', r'\1', expr_str)      # Remove [ ]
    expr_str = re.sub(r'^\\\(\s*(.+?)\s*\\\)$', r'\1', expr_str)  # Remove \( \)
    # expr_str = re.sub(r'^\(\s*(.+?)\s*\)$', r'\1', expr_str)      # Remove ( )
    return expr_str

def clean_expr_str(expr_str):

    expr_str = re.sub(r'^\$\$\s*(.+?)\s*\$\$$', r'\$\1\$', expr_str)      # replace  $$ $$ -> $ $
    expr_str = re.sub(r'^\$\$\n\s*(.+?)\s*\n\$\$$', r'\$\1\$', expr_str)      # replace  $$ $$ -> $ $
    
    expr_str = (
        expr_str.replace(" . ", ".")
        .replace(". ", ".")
        .replace("**", "^")
        .replace("\\pm", "")
        .replace("*", "\\times ")
        .replace("\\\\", "\\")
        .replace("\\ne ", "\\neq ")
        .replace("!=", "\\neq")
        .replace(">=", "\\ge")
        .replace("<=", "\\le")
        .replace("≠", "\\neq")
        .replace("dfrac", "frac")
        .replace("tfrac", "frac")
        .replace("\\$", "")
        .replace("$", "")
        .replace("\\%", "")
        .replace("%", "")
        .replace("\\!", "")
        .replace("^\circ", "\\times \\pi / 180")
        .replace("//", "/")
        .replace('"', "")
        # .replace(",", "") # TODO
    )
    # expr_str = re.sub(r"\^\s(.*)", r"\^\s{\1}", expr_str) # replacement causing error 
    expr_str = re.sub(r"\\+", r"\\", expr_str)
    expr_str = re.sub(r"\^\s?\((.*?)\)", r"^{\1}", expr_str)
    expr_str = re.sub(r"\\frac\s?(\d)\s?(\d+)", r"\\frac{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\log_\s?(\d)\s?(\d+)", r"\\log_{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\frac\s?{(.*?)}\s?(\d)", r"\\frac{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\frac\s?(\d)\s?{(.*?)}", r"\\frac{\1}{\2}", expr_str)
    expr_str = re.sub(r"\\sqrt\s?(\d)", r"\\sqrt{\1}", expr_str)
    expr_str = re.sub(r"sqrt\s?\((\d+)\)", r"\\sqrt{\1}", expr_str)
    expr_str = re.sub(r"sqrt\s?\((.*?)\)", r"\\sqrt{\1}", expr_str)
    expr_str = expr_str.replace(" sqrt", "\\sqrt")
    expr_str = (
        expr_str.replace("\\left", "").replace("\\right.", "").replace("\\right", "")
    )
    return expr_str

def my_parse_latex(expr_str):
    expr_str = expr_str.replace("dfrac", "frac")
    expr = parse_latex(expr_str)
    if "\\pi" in expr_str:
        expr = expr.subs({sp.Symbol("pi"): sp.pi})
    expr = expr.subs({sp.Symbol("i"): sp.I})
    return expr

def parse_latex_answer(sample):
    if isinstance(sample, int) or isinstance(sample, float):
        sample = str(sample)
    #     return sample
    sample = clean_expr_str(sample)
    try:
        expr = my_parse_latex(sample)
    except:
        print("[parse failed]", sample)
        return ""
    return expr

def looks_like_label(text: str) -> bool:
    """Check if the string looks like a label (e.g., 'A.', 'Q1.', 'C. 7')"""
    label_patterns = [
        r'^[A-Z]\.\s*\d*$',  # Matches "A.", "B. 7"
        r'^[A-Z]\d+\.',      # Matches "Q1.", "P2."
        r'^\d+\.',           # Matches "1.", "42."
        r'^[A-Z]\s*\d+$',    # Matches "C 7", "A 42"
    ]
    return any(bool(re.match(pattern, text.strip())) for pattern in label_patterns)


def extract_label_content(text: str) -> tuple[bool, str, str]:
    """
    Check if the string looks like a label and extract content after it.
    
    Args:
        text (str): Input text to check
    
    Returns:
        tuple[bool, str, str]: (is_label, label, content)
            - is_label: Whether the text matches a label pattern
            - label: The label part if found, empty string if not
            - content: The content after label if found, empty string if not
    """
    text = text.strip()
    patterns = [
        # pattern, group names for (label, content)
        (r'^([A-Z]\.)\s*(.+)?$', 1, 2),         # Matches "A.", "B. sin(x)"
        (r'^([A-Z]\d+\.)\s*(.+)?$', 1, 2),      # Matches "Q1.", "P2. anything"
        (r'^(\d+\.)\s*(.+)?$', 1, 2),           # Matches "1.", "42. anything"
        (r'^([A-Z]\s*\d+)\s*(.+)?$', 1, 2),     # Matches "C 7", "A 42 anything"
        (r'^\(([A-Z])\)\s*(.+)?$', 1, 2),       # Matches "(A)", "(B) content"
        (r'^([A-Z]\))\s*(.+)?$', 1, 2),         # Matches "A)", "B) content"
        (r'^\[([A-Z])\]\s*(.+)?$', 1, 2),       # Matches "[A]", "[B] content"
    ]
    
    for pattern, label_group, content_group in patterns:
        match = re.match(pattern, text)
        if match:
            label = match.group(label_group).strip()
            # Remove any trailing parenthesis or bracket from the label
            label = re.sub(r'[\)\]]$', '', label)
            content = match.group(content_group).strip() if match.group(content_group) else ""
            return True, label, content
            
    return False, "", ""

def has_valid_math_chars(text: str) -> bool:
    """Check if string contains valid mathematical characters."""
    # Define pattern for valid math expression
    # This includes: numbers, operators, letters (for variables), parentheses, and whitespace
    # print('checking has_valid_math_chars():', text)
    math_pattern = r'^[a-zA-Z0-9\s\+\-\*/\^\(\)\=\.\,\[\]\{\}]+$'
    return bool(re.match(math_pattern, text))

def has_mathematical_structure(expr_str: str) -> bool:
    """Check if string has basic mathematical structure."""
    # Must contain either a number or an operator (excluding =)
    has_number = bool(re.search(r'\d', expr_str))
    has_operator = any(op in expr_str for op in ['+', '-', '*', '/', '^', '**'])
    # If it has a number, we don't require an operator (e.g., "2" is valid)
    # If it has an operator, we don't require a number (e.g., "x + y" is valid)
    return has_number or has_operator

def contains_text_command(text: str) -> bool:
    """
    Check if the string contains LaTeX text commands like \text{}, \textbf{}, \textit{}, etc.
    Returns True if any LaTeX text command is found.
    Handles expressions inside \( \), $...$ and other LaTeX delimiters.
    """
    return '\text' in text or '\\\text' in text
    # text = text.replace('\text', '\\text')
    # # First clean up any escaped backslashes
    # text = text.replace("\\\\", "\\")
    
    # Updated pattern to detect LaTeX text formatting commands
    # Matches:
    # - \text{...}, \textbf{...}, \textit{...}, etc.
    # - Expressions inside \( ... \), $...$, etc.
    # pattern = r'\\(?:text[a-zA-Z]*|[a-zA-Z]+)\s*\{'
    
    # # Match any LaTeX command in the text
    # return bool(re.search(pattern, text))

def is_just_text(expr):
    """Check if expression is just a sequence of multiplied variables."""
    if isinstance(expr, (Number, int, float)):
        return False
    if isinstance(expr, Symbol):
        return True
    if hasattr(expr, 'is_Add') or hasattr(expr, 'is_Pow'):
        return False
    if hasattr(expr, 'is_Mul'):
        # Check if any part is a number or operation
        has_number = any(isinstance(arg, (Number, int, float)) for arg in expr.args)
        has_operation = any(hasattr(arg, 'is_Add') or hasattr(arg, 'is_Pow') for arg in expr.args)
        return not (has_number or has_operation) and all(isinstance(arg, Symbol) for arg in expr.args)
    return False

def has_text_indicators(text: str) -> bool:
    """
    Check if the string contains indicators of natural language text.
    Returns True if the string appears to contain natural language.
    """
    # Common words that indicate text content
    text_indicators = {
        # Logical connectors
        'therefore', 'thus', 'hence', 'consequently', 'accordingly', 'so',
        'because', 'since', 'due to', 'as a result', 'follows that',
        
        # Mathematical discourse
        'where', 'when', 'if', 'then', 'given', 'suppose', 'let',
        'consider', 'assume', 'assuming', 'exists', 'provided',
        'defined', 'denote', 'denoted', 'implies', 'proves', 'proved',
        'showing', 'shows', 'follows', 'following', 'holds', 'satisfies',
        
        # Common articles and prepositions
        'the', 'a', 'an', 'and', 'or', 'as', 'by', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'without', 'from', 'into', 'onto',
        'under', 'over', 'above', 'below', 'between', 'among',
        
        # Mathematical description
        'solution', 'equation', 'function', 'variable', 'value',
        'expression', 'formula', 'proof', 'theorem', 'lemma',
        'corollary', 'proposition', 'definition', 'identity',
        'condition', 'boundary', 'initial', 'final', 'system',
        
        # Process words
        'solve', 'solving', 'solved', 'calculate', 'calculating',
        'computed', 'computing', 'derive', 'derived', 'deriving',
        'find', 'finding', 'found', 'determine', 'determined',
        
        # Quantifiers and descriptors
        'all', 'every', 'any', 'some', 'none', 'no', 'each',
        'many', 'few', 'several', 'must', 'should', 'can', 'may',
        'possible', 'impossible', 'necessary', 'sufficient',
        
        # Result description
        'answer', 'result', 'conclusion', 'solution', 'yields',
        'obtains', 'gives', 'produces', 'leads', 'reduces',
        'simplifies', 'equals', 'equivalent', 'same', 'different',
        
        # Math relationships
        'greater', 'less', 'equal', 'unequal', 'equivalent',
        'approximately', 'about', 'roughly', 'exactly', 'precisely',
        'minimum', 'maximum', 'extremum', 'optimal', 'optimization'
    }
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Check for common punctuation that suggests sentences
    has_sentence_punctuation = any(p in text for p in ['. ', '; ', '! ', '? '])
    
    # Check for presence of common words
    has_common_words = any(f' {word} ' in f' {text_lower} ' for word in text_indicators)
    
    # Check for multiple words (more than 2) by counting spaces
    word_count = len(text.split())
    has_many_words = word_count > 2
    
    return has_sentence_punctuation or has_common_words or has_many_words

def split_math_and_text(text: str) -> list[tuple[str, bool]]:
    """
    Split a string into segments of math and text.
    Returns a list of (segment, is_math) tuples.
    Math segments are those enclosed in common math delimiters.
    """
    # Common math delimiters
    delimiters = [
        (r'\[', r'\]'),
        (r'\(', r'\)'),
        ('$', '$'),
    ]
    
    # Initialize result list and current position
    segments = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # Find the next math segment
        next_math_start = text_length
        next_delimiter_pair = None
        
        for start_delim, end_delim in delimiters:
            start_pos = text.find(start_delim, current_pos)
            if start_pos != -1 and start_pos < next_math_start:
                next_math_start = start_pos
                next_delimiter_pair = (start_delim, end_delim)
        
        if next_delimiter_pair:
            # Add text segment before math if exists
            if current_pos < next_math_start:
                text_segment = text[current_pos:next_math_start].strip()
                if text_segment:
                    segments.append((text_segment, False))
            
            # Find the end of math segment
            start_delim, end_delim = next_delimiter_pair
            math_start = next_math_start + len(start_delim)
            math_end = text.find(end_delim, math_start)
            
            if math_end == -1:
                # No closing delimiter found, treat rest as text
                segments.append((text[current_pos:], False))
                break
            
            # Add math segment
            math_segment = text[math_start:math_end].strip()
            if math_segment:
                segments.append((math_segment, True))
            
            current_pos = math_end + len(end_delim)
        else:
            # No more math segments, add remaining text
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                segments.append((remaining_text, False))
            break
    
    return segments


# from wrapt_timeout_decorator import *
# @timeout(5)
def is_valid_math(text: str, allow_latex: bool = True) -> tuple[bool, str, str]:
    """
    Check if a string is a valid mathematical expression or equation.
    
    Args:
        text (str): The string to validate
        allow_latex (bool): Whether to attempt parsing LaTeX syntax
    
    Returns:
        tuple[bool, str, str]: (is_valid, type, parsed_expression)
            - is_valid: Whether the string is valid math
            - type: "expression", "equation", or "invalid"
            - parsed_expression: String representation of parsed math if valid, error message if invalid
    """
    # Strip whitespace and replace ^ with **
    text = text.strip()
    
    # Early validation checks
    if not text:
        return False, "invalid", "Empty input"



    # Check if it looks like a label
    if looks_like_label(text):
        return False, "invalid", "Appears to be a label or numbering"
    
    # Check for text indicators
    if has_text_indicators(text):
        return False, "invalid", "Contains natural language text"
    
    if is_number(text):
        return True, "numbers", text
    
    if contains_text_command(text):
        return False, "latex text", "Contains \\text{} in expression"
    
    # Skip LaTeX validation if it looks like LaTeX
    if not text.startswith('\\') and not text.startswith('$'):
        if not has_valid_math_chars(text):
            return False, "invalid", "Contains invalid characters for mathematical expression"
        if not has_mathematical_structure(text):
            return False, "invalid", "Lacks mathematical structure (no numbers or operators)"
    
    
    # if not in latex mode, use "**" rather than "^"
    text = text.replace('^', '**')
    
    # Configure parsing transformations
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    # First try parsing as normal math expression
    try:
        # Check if it's an equation (contains =)
        if '=' in text:
            # Split on first equals sign
            left, right = text.split('=', 1)
            try:
                left_expr = parse_expr(left.strip(), transformations=transformations)
                right_expr = parse_expr(right.strip(), transformations=transformations)
                eq = Eq(left_expr, right_expr)
                return True, "equation", str(eq)
            except (TokenError, SyntaxError) as e:
                # print(f'debug error 1:', text)
                pass
        else:
            # Try parsing as expression
            expr = parse_expr(text, transformations=transformations)
            
            # Only reject if it's purely text multiplication
            if is_just_text(expr):
                return False, "invalid", "Expression appears to be just text"
                
            return True, "expression", str(expr)
        
            
    except Exception as e:
        if not allow_latex:
            return False, "invalid", f"Parse error : {str(e)}"

    # If normal parsing failed and LaTeX is allowed, try parsing as LaTeX then!
    if allow_latex:
        try:
            latex_text = clean_expr_str(text) # pre-processing first
            # # Remove $ signs if present
            latex_text = latex_text.strip('$')
            
            # Check if it's a LaTeX equation
            if '=' in latex_text:
                left, right = latex_text.split('=', 1)
                try:
                    left_expr = parse_latex(left.strip())
                    right_expr = parse_latex(right.strip())
                    eq = Eq(left_expr, right_expr)
                    return True, "latex equation", str(eq)
                except Exception as e:
                    return False, "invalid", f"LaTeX equation parse error: {str(e)}"
            else:
                expr = parse_latex(latex_text)
                return True, "latex expression", str(expr)
        except Exception as e:
            return False, "invalid", f"LaTeX parse error: {str(e)}"

    
    return False, "invalid", "Could not parse as math expression or equation"

