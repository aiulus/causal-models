import numpy as np
from utils import io


def generate_linear_function(parents, coeffs):
    """Return a linear structural equation as a lambda string (noise added externally)."""
    terms = [f"{coeffs[i]} * {parent}" for i, parent in enumerate(parents)]
    expr = " + ".join(terms) if terms else "0"
    return f"lambda {', '.join(parents)}: {expr}"


def generate_polynomial_function(parents, coeffs, degrees):
    """Return a polynomial structural equation as a lambda string (noise added externally)."""
    terms = [
        f"{coeffs[i]} * {parent} ** {degrees[i]}"
        for i, parent in enumerate(parents)
    ]
    expr = " + ".join(terms) if terms else "0"
    return f"lambda {', '.join(parents)}: {expr}"


def generate_functions(graph, noise_vars, funct_type='linear'):
    """
    Create a structural equation f_i for each node using specified type.
    Noise is handled externally in sampler.
    """
    functions = {}
    coeff_set = io.config['COEFFICIENTS']
    max_degree = io.config['MAX_POLYNOMIAL_DEGREE']

    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        coeffs = np.random.choice(coeff_set, size=len(parents))

        if funct_type == 'linear':
            functions[node] = generate_linear_function(parents, coeffs)

        elif funct_type == 'polynomial':
            degrees = np.random.randint(1, max_degree + 1, size=len(parents))
            functions[node] = generate_polynomial_function(parents, coeffs, degrees)

        else:
            raise ValueError(f"Unsupported function type: {funct_type}")

    return functions


def parse_functions(func_dict):
    """
    Convert dictionary of function strings to actual callables using eval.
    (Unsafe - replaceable with AST parser in future refactor.)
    """
    return {key: eval(func_str) for key, func_str in func_dict.items()}
