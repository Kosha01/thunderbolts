from flask import Flask, render_template, request, jsonify, flash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, FloatField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
from scipy import stats
import re
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')  # Change in production
csrf = CSRFProtect(app)

class ProblemForm(FlaskForm):
    problem_text = StringField('Problem Description', validators=[DataRequired()])
    submit = SubmitField('Solve Problem')

def extract_number(text, pattern):
    """Helper function to extract numbers from text"""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def identify_distribution(text):
    """
    Identify the statistical distribution from the problem text.
    Returns: tuple (distribution_name, parameters_dict, target_value)
    """
    text = text.lower()
    
    # Extract target value (k or x) if present
    target = None
    exactly_k = re.search(r'exactly\s+(\d+)', text)
    value_x = re.search(r'x\s*=\s*(\d*\.?\d+)', text)
    if exactly_k:
        target = int(exactly_k.group(1))
    elif value_x:
        target = float(value_x.group(1))
    
    # Binomial Distribution
    if any(word in text for word in ['binomial', 'success', 'failure', 'trials']):
        n = extract_number(text, r'(\d+)\s+trials')
        p = extract_number(text, r'probability\s+of\s+success\s+is\s+(\d*\.?\d+)')
        if n is not None and p is not None:
            return 'binomial', {'n': int(n), 'p': p}, target
    
    # Poisson Distribution
    if any(word in text for word in ['poisson', 'rate', 'average rate']):
        lambda_val = extract_number(text, r'rate\s+of\s+(\d*\.?\d+)')
        if lambda_val is not None:
            return 'poisson', {'lambda': lambda_val}, target
    
    # Normal Distribution
    if any(word in text for word in ['normal', 'gaussian']):
        mean = extract_number(text, r'mean\s+of\s+(\d*\.?\d+)')
        std = extract_number(text, r'standard\s+deviation\s+of\s+(\d*\.?\d+)')
        if mean is not None and std is not None:
            return 'normal', {'mean': mean, 'std': std}, target
    
    return 'unknown', {}, None

def calculate_probability(distribution, params, target_value=None):
    """
    Calculate probability based on the identified distribution and parameters.
    Returns: tuple (result, steps)
    """
    steps = []
    result = None
    
    try:
        if distribution == 'binomial':
            n, p = params['n'], params['p']
            steps.append(f"Using Binomial Distribution with n={n} trials and p={p}")
            steps.append(f"P(X = k) = C(n,k) * p^k * (1-p)^(n-k)")
            
            k = target_value if target_value is not None else n//2
            result = stats.binom.pmf(k, n, p)
            steps.append(f"For k = {k}:")
            steps.append(f"P(X = {k}) = C({n},{k}) * {p}^{k} * (1-{p})^({n}-{k})")
            steps.append(f"Probability = {result:.4f}")
        
        elif distribution == 'poisson':
            lambda_val = params['lambda']
            steps.append(f"Using Poisson Distribution with λ={lambda_val}")
            steps.append(f"P(X = k) = (λ^k * e^(-λ)) / k!")
            
            k = target_value if target_value is not None else int(lambda_val)
            result = stats.poisson.pmf(k, lambda_val)
            steps.append(f"For k = {k}:")
            steps.append(f"P(X = {k}) = ({lambda_val}^{k} * e^(-{lambda_val})) / {k}!")
            steps.append(f"Probability = {result:.4f}")
        
        elif distribution == 'normal':
            mean, std = params['mean'], params['std']
            steps.append(f"Using Normal Distribution with μ={mean} and σ={std}")
            steps.append(f"Using standard normal distribution formula:")
            steps.append(f"Z = (X - μ) / σ")
            
            x = target_value if target_value is not None else mean
            result = stats.norm.pdf(x, mean, std)
            z_score = (x - mean) / std
            steps.append(f"For x = {x}:")
            steps.append(f"Z = ({x} - {mean}) / {std} = {z_score:.4f}")
            steps.append(f"Probability density at x={x} is {result:.4f}")
        
        return result, steps
    
    except Exception as e:
        return None, [f"Error in calculation: {str(e)}"]

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ProblemForm()
    result = None
    steps = None
    distribution = None
    error = None
    
    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                problem_text = form.problem_text.data
                distribution, params, target = identify_distribution(problem_text)
                
                if distribution == 'unknown':
                    error = "Could not identify the distribution type or extract necessary parameters. Please check your input format."
                else:
                    result, steps = calculate_probability(distribution, params, target)
                    if result is None:
                        error = "Error occurred during calculation. Please check your input values."
            
            except Exception as e:
                error = f"An error occurred: {str(e)}"
        else:
            error = "Invalid form submission. Please try again."
    
    return render_template('index.html', form=form, result=result, 
                         steps=steps, distribution=distribution, error=error)

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', 
                         form=ProblemForm(),
                         error="An internal server error occurred. Please try again."), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html',
                         form=ProblemForm(),
                         error="Page not found. Please check the URL."), 404

if __name__ == '__main__':
    app.run(debug=True) 