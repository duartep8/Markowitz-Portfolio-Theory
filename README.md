# Markowitz Portfolio Theory Implementation in Python

Welcome to this implementation of Harry Markowitz's Modern Portfolio Theory (MPT) in Python! 😊

This project provides a practical demonstration of how to construct an efficient frontier, identify the Minimum Variance Portfolio (MVP), and determine the optimal Tangency Portfolio. The goal is to find the optimal allocation of assets that maximizes the expected return $E[R_p]$ for a given level of risk (volatility, $\sigma_p$).

## 📊 Features

* Calculates the **Efficient Frontier** for a basket of assets.
* Identifies and displays the characteristics of the **Minimum Variance Portfolio (MVP)**.
* Determines the **Tangency Portfolio** (also known as the Optimal Risky Portfolio).
* Plots the **Capital Market Line (CML)**.
* Generates a comprehensive plot visualizing all key components of the theory.

## 🚀 Getting Started

Please follow these simple steps to set up the project and see the results.

### Prerequisites

Before running the scripts, you need to create a dedicated folder for the financial data.

* In the root directory of this project, create a new folder and name it `data`.

### Installation & Setup

The project is designed for ease of use. Instead of manual commands, you can simply run the provided Python scripts.

1.  **Install Dependencies**
    Run the `install_dependencies.py` script. This will automatically install all the required Python libraries.
    ```bash
    python install_dependencies.py
    ```

2.  **Download Financial Data**
    Next, run the `download_data.py` script. This will fetch the necessary historical stock data and save it into the `data` folder you created.
    ```bash
    python download_data.py
    ```

## 📈 Usage

Once the setup is complete, you are ready to run the main analysis. (some plots need to run the plot_functions.py directly)

Execute the `main.py` script:
```bash
python main.py