# Statistical Distribution Calculator

A Flask web application that helps users solve probability problems by automatically identifying the appropriate statistical distribution, extracting parameters, and providing step-by-step solutions.

## Features

- Supports multiple statistical distributions:
  - Binomial Distribution
  - Poisson Distribution
  - Normal Distribution
  - (More distributions to be added)
- Automatic distribution identification from problem text
- Parameter extraction from natural language
- Step-by-step solution display
- Clean and modern user interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your probability problem in natural language. For example:
   - "In a binomial distribution with 10 trials and probability of success is 0.3, what is the probability of exactly 3 successes?"
   - "In a Poisson distribution with a rate of 2.5 events per hour, what is the probability of exactly 3 events?"
   - "For a normal distribution with mean of 70 and standard deviation of 5, what is the probability density at x=75?"

2. Click "Solve Problem" to get:
   - Identified distribution
   - Step-by-step solution
   - Final numerical result

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. #   t h u n d e r b o l t s  
 #   t h u n d e r b o l t s  
 