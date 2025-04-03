from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy import stats
import re

app = Flask(__name__)

def extract_number(text):
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return [float(num) for num in numbers]

def identify_distribution(text):
    text = text.lower()
    if 'binomial' in text:
        numbers = extract_number(text)
        if len(numbers) >= 2:
            n = int(numbers[0])
            p = numbers[1]
            return 'binomial', {'n': n, 'p': p}
    elif 'poisson' in text:
        numbers = extract_number(text)
        if numbers:
            lambda_param = numbers[0]
            return 'poisson', {'lambda': lambda_param}
    elif 'normal' in text:
        numbers = extract_number(text)
        if len(numbers) >= 2:
            mu = numbers[0]
            sigma = numbers[1]
            return 'normal', {'mu': mu, 'sigma': sigma}
    return None, None

def calculate_probability(dist_type, params, target=None):
    if not target:
        target = extract_number(params.get('target', '0'))[0]

    if dist_type == 'binomial':
        n, p = params['n'], params['p']
        prob = stats.binom.pmf(target, n, p)
        steps = [
            f"Using Binomial Distribution with n={n}, p={p}",
            f"P(X = {target}) = C({n},{target}) * {p}^{target} * (1-{p})^({n}-{target})",
            f"Probability = {prob:.4f}"
        ]
        return prob, steps

    elif dist_type == 'poisson':
        lambda_param = params['lambda']
        prob = stats.poisson.pmf(target, lambda_param)
        steps = [
            f"Using Poisson Distribution with λ={lambda_param}",
            f"P(X = {target}) = (e^(-{lambda_param}) * {lambda_param}^{target}) / {target}!",
            f"Probability = {prob:.4f}"
        ]
        return prob, steps

    elif dist_type == 'normal':
        mu, sigma = params['mu'], params['sigma']
        prob = stats.norm.pdf(target, mu, sigma)
        steps = [
            f"Using Normal Distribution with μ={mu}, σ={sigma}",
            f"P(X = {target}) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))",
            f"Probability = {prob:.4f}"
        ]
        return prob, steps

    return None, ["Distribution not recognized"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    problem_text = data.get('problem_text', '')
    
    dist_type, params = identify_distribution(problem_text)
    if not dist_type:
        return jsonify({
            'error': 'Could not identify distribution type',
            'steps': ['Please provide a valid probability problem']
        })

    target = None
    if 'exactly' in problem_text.lower():
        numbers = extract_number(problem_text)
        if len(numbers) > 2:  # Assuming the last number is the target
            target = int(numbers[-1])

    prob, steps = calculate_probability(dist_type, params, target)
    
    if prob is None:
        return jsonify({
            'error': 'Could not calculate probability',
            'steps': steps
        })

    return jsonify({
        'distribution': dist_type,
        'parameters': params,
        'probability': float(prob),
        'steps': steps
    })

if __name__ == '__main__':
    app.run(debug=True) 