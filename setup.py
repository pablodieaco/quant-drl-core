from setuptools import find_packages, setup

setup(
    name="quant_drl",
    version="0.1.0",
    description="Deep Reinforcement Learning for Financial Portfolio Management",
    author="Pablo Diego Acosta",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "gym",
        "matplotlib",
        "stable-baselines3",
        "scikit-learn",
        "loguru",
        "tqdm",
        "tensorflow",
        "torch",
        "plotly",
        "seaborn",
        "yfinance",
        "ta",
        "pandas_market_calendars",
    ],
    include_package_data=True,
)
