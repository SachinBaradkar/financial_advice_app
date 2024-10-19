import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Machine Learning Model Training
def train_models():
    X = np.array([[100000], [300000], [500000], [800000]])
    y = np.array([
        [0.08, 0.05, 0.0],  # Lower class
        [0.10, 0.07, 0.05], # Lower-middle class
        [0.12, 0.10, 0.10], # Middle class
        [0.14, 0.12, 0.15]  # Upper-middle class
    ])
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor()
    }
    
    for model in models.values():
        model.fit(X, y)
    
    return models

def predict_rates(models, salary):
    predictions = {name: model.predict(np.array([[salary]]))[0] for name, model in models.items()}
    return predictions

# Tax Calculation Function
def calculate_tax(salary, regime="new"):
    if regime == "new":
        if salary <= 300000:
            tax = 0
        elif salary <= 700000:
            tax = 0.05 * (salary - 300001)
        elif salary <= 1000000:
            tax = 0.10 * (salary - 700001)
        else:
            tax = 0.15 * (salary - 1000001)
    return tax

# Function to calculate financial plan
def calculate_financial_plan(salary, models, regime, family_members, yearly):
    predictions = predict_rates(models, salary)
    
    # Use one model's predictions for rates
    savings_rate, investment_rate, tax_rate = predictions["Linear Regression"]  # or Decision Tree

    tax = calculate_tax(salary, regime) if yearly else 0

    # Adjust savings and investment rates based on family members
    adjusted_savings_rate = savings_rate - (family_members - 1) * 0.002
    adjusted_investment_rate = investment_rate - (family_members - 1) * 0.002

    # Ensure rates do not go below zero
    adjusted_savings_rate = max(adjusted_savings_rate, 0)
    adjusted_investment_rate = max(adjusted_investment_rate, 0)

    # Expenses: Adjust based on family members
    base_expense_rate = 0.5
    additional_expense_per_member = 0.1
    adjusted_expense_rate = base_expense_rate + (family_members - 1) * additional_expense_per_member

    # Calculate savings, investment, expenses
    savings = salary * adjusted_savings_rate
    investment = salary * adjusted_investment_rate
    expenses = salary * adjusted_expense_rate

    total_used = savings + investment + tax + expenses

    if total_used != salary:
        difference = salary - total_used
        adjustment_factor = difference / (adjusted_savings_rate + adjusted_investment_rate + adjusted_expense_rate)
        savings += adjustment_factor * adjusted_savings_rate
        investment += adjustment_factor * adjusted_investment_rate
        expenses += adjustment_factor * adjusted_expense_rate

    expenses = max(min(expenses, salary), 0)

    return savings, investment, tax, expenses

# SIP Calculator
def sip_calculator(monthly_investment, rate_of_return, investment_period):
    rate_of_return /= 100
    months = investment_period * 12
    monthly_rate = rate_of_return / 12
    future_value = monthly_investment * ((1 + monthly_rate) ** months - 1) / monthly_rate * (1 + monthly_rate)
    return future_value

# Plotting Function
def plot_rates(predictions):
    labels = list(predictions.keys())
    savings = [pred[0] for pred in predictions.values()]
    investments = [pred[1] for pred in predictions.values()]

    x = np.arange(len(labels))
    
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, savings, 0.4, label='Savings Rate')
    ax.bar(x + 0.2, investments, 0.4, label='Investment Rate')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Rate')
    ax.set_title('Savings and Investment Rates by Model')
    ax.legend()
    
    st.pyplot(fig)

# Financial Goals Section
def financial_goals():
    goal_amount = st.number_input("Enter your financial goal amount (₹):", min_value=0.0, step=10000.0)
    years_to_save = st.number_input("Enter the number of years to achieve this goal:", min_value=1, max_value=30, step=1)
    monthly_investment = goal_amount / (years_to_save * 12)  # Simple calculation

    st.write(f"You need to save/invest approximately *₹{monthly_investment:,.2f}* per month to reach your goal.")

# Main function to run the app
def main():
    st.set_page_config(page_title="Smart Financial Advice", page_icon="💼", layout="centered")
    
    st.title("💼 Smart Financial Advice ")
    st.markdown("*Balance Your Budget, Build Your Wealth! Spend Smart, Invest Smarter!*")

    salary = st.number_input("Enter your monthly salary (₹):", min_value=0.0, step=1000.0, max_value=800000.0)
    family_members = st.slider("Number of family members:", min_value=1, max_value=7, value=1)
    view_option = st.radio("Select view:", ('Monthly', 'Yearly'))
    yearly = view_option == 'Yearly'
    tax_regime = st.selectbox("Select Tax Regime:", ('New', ''))

    models = train_models()

    if salary > 0:
        multiplier = 12 if yearly else 1
        display_salary = salary * multiplier
        savings, investment, tax, expenses = calculate_financial_plan(display_salary, models, tax_regime.lower(), family_members, yearly)
        
        st.subheader(f"💡 Financial Breakdown ({view_option})")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Savings", value=f"₹{savings:,.2f}")
            st.metric(label="Investments", value=f"₹{investment:,.2f}")
        with col2:
            st.metric(label="Taxes", value=f"₹{tax:,.2f}" if yearly else "N/A")
            st.metric(label="Expenses", value=f"₹{expenses:,.2f}")

        st.subheader("📈 Advice Summary")
        st.markdown(f"""
        1. *Savings*: Set aside the above amount as savings for emergencies and future needs.
        2. *Investments*: Invest the above amount in investment options like Fixed Deposits, PPF, LIC, Gold, and Mutual Funds.
        3. *Taxes*: Based on the new tax regime, reserve your income for taxes (applicable to yearly view only).
        4. *Expenses*: Allocate the remaining amount for essential expenses such as food, housing, transportation, education, and healthcare.
        5. *Tips: **A. Grow Your Wealth, Don’t Overspend.*    
                 *B. Invest Today, Enjoy Tomorrow – Keep Expenses in Check.*         
        """)

        # Display final matching of salary and output
        total_used = savings + investment + tax + expenses
        st.write(f"*Total Salary*: ₹{display_salary:,.2f}")
        st.write(f"*Total Used*: ₹{total_used:,.2f}")

        if abs(total_used - display_salary) > 1:
            st.warning("The total allocations do not exactly match your salary. Please review the details.")
        else:
            st.success("The allocations are balanced with your salary.")

        # SIP Calculator section
        st.subheader("📊 SIP Calculator")
        monthly_investment = st.number_input("Monthly Investment Amount (₹):", min_value=500.0, step=500.0)
        rate_of_return = st.slider("Expected Annual Rate of Return (%):", min_value=1.0, max_value=30.0, step=0.5)
        investment_period = st.slider("Investment Period (Years):", min_value=1, max_value=30)

        if monthly_investment > 0:
            future_value = sip_calculator(monthly_investment, rate_of_return, investment_period)
            st.metric(label="Future Value of Investment", value=f"₹{future_value:,.2f}")

      

        # Financial Goals Section
        st.subheader("🎯 Financial Goals")
        financial_goals()

if __name__ == "__main__":
    main()
