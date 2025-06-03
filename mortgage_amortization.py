import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_mortgage_payment(
    principal, annual_rate, loan_term_years, payment_frequency_per_year
):
    """Calculates the fixed periodic mortgage payment."""
    if annual_rate == 0:  # Handle 0% interest rate to avoid division by zero
        if loan_term_years == 0:  # Avoid division by zero if loan term is also zero
            return 0.0
        return principal / (loan_term_years * payment_frequency_per_year)

    periodic_rate = annual_rate / payment_frequency_per_year
    n_payments = loan_term_years * payment_frequency_per_year

    # Mortgage payment formula: M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
    payment = (
        principal
        * (periodic_rate * (1 + periodic_rate) ** n_payments)
        / ((1 + periodic_rate) ** n_payments - 1)
    )
    return payment


def generate_mortgage_amortization_schedule(
    principal, annual_rate, loan_term_years, payment_frequency_per_year
):
    """Generates a mortgage amortization schedule."""
    periodic_rate = annual_rate / payment_frequency_per_year
    n_payments = loan_term_years * payment_frequency_per_year
    fixed_payment = calculate_mortgage_payment(
        principal, annual_rate, loan_term_years, payment_frequency_per_year
    )

    current_balance = principal
    schedule_data = []

    for i in range(1, int(n_payments) + 1):
        beginning_balance = current_balance

        # Calculate interest paid
        interest_paid = beginning_balance * periodic_rate

        # Calculate principal paid
        principal_paid = fixed_payment - interest_paid

        # Adjust for the very last payment to ensure the balance goes to exactly zero
        # This handles potential floating-point inaccuracies
        if i == n_payments:
            principal_paid = (
                beginning_balance  # Principal paid should be exactly what's left
            )
            interest_paid = (
                fixed_payment - principal_paid
            )  # Interest is what remains of the fixed payment
            # Ensure interest is not negative due to slight rounding issues if principal_paid was adjusted up
            if interest_paid < 0:
                interest_paid = 0
            ending_balance = 0.0  # Force ending balance to zero
        else:
            ending_balance = beginning_balance - principal_paid
            # If due to rounding, the balance briefly dips below zero before the last payment,
            # adjust the principal for that payment to bring it to zero, and the next payment should be regular
            if ending_balance < 0 and (n_payments - i) > 1:  # Not the final payment
                principal_paid = beginning_balance  # Pay off remaining balance
                interest_paid = fixed_payment - principal_paid
                if interest_paid < 0:
                    interest_paid = 0
                ending_balance = 0.0

        schedule_data.append(
            {
                "Payment No.": i,
                "Beginning Balance": beginning_balance,
                "Monthly Payment": fixed_payment,  # This remains the constant calculated payment for most periods
                "Interest Paid": interest_paid,
                "Principal Paid": principal_paid,
                "Ending Balance": ending_balance,
            }
        )
        current_balance = ending_balance

    df = pd.DataFrame(schedule_data)

    # Final adjustment to ensure total principal paid exactly equals original principal
    # This is a cumulative adjustment for any tiny floating point errors over many periods
    if not df.empty:
        total_principal_paid_actual = df["Principal Paid"].sum()
        if not np.isclose(
            total_principal_paid_actual, principal, atol=0.01
        ):  # Check if significantly different from principal
            diff = principal - total_principal_paid_actual
            # Distribute the small difference to the last principal payment
            df.loc[df.index[-1], "Principal Paid"] += diff
            # Recalculate interest for the last payment based on adjusted principal to maintain fixed_payment
            df.loc[df.index[-1], "Interest Paid"] = (
                df.loc[df.index[-1], "Monthly Payment"]
                - df.loc[df.index[-1], "Principal Paid"]
            )
            # Ensure ending balance is zero for the last period
            df.loc[df.index[-1], "Ending Balance"] = 0.0

    return df


# --- Streamlit Dashboard Layout ---
st.set_page_config(layout="wide", page_title="Mortgage Amortization Dashboard")

st.title("Mortgage Amortization Schedule Comparison")

st.markdown("""
This dashboard allows you to compare the amortization schedules of two mortgage loans.
Enter the parameters for each loan below.
""")

# --- Mortgage Input Section ---
col1, col2 = st.columns(2)

mortgage_params = {}

# Default values for easier comparison
default_loan_amount = 300000.0
default_annual_rate_1 = 0.065  # 6.5%
default_annual_rate_2 = 0.07  # 7.0%
default_term_1 = 30  # 30 years
default_term_2 = 15  # 15 years

for i, col in enumerate([col1, col2]):
    with col:
        st.header(f"Mortgage {i + 1} Parameters")
        # Ensure unique keys for widgets
        mortgage_params[f"mortgage_{i + 1}_principal"] = st.number_input(
            f"Loan Amount (Mortgage {i + 1})",
            value=default_loan_amount,
            min_value=1000.0,
            step=1000.0,
            format="%.2f",
            key=f"principal_{i + 1}_input",
        )
        mortgage_params[f"mortgage_{i + 1}_rate"] = st.number_input(
            f"Annual Interest Rate (Mortgage {i + 1}, e.g., 0.06 for 6%)",
            value=default_annual_rate_1 if i == 0 else default_annual_rate_2,
            min_value=0.0,
            max_value=0.20,
            format="%.4f",
            key=f"rate_{i + 1}_input",
        )
        mortgage_params[f"mortgage_{i + 1}_term"] = st.number_input(
            f"Loan Term in Years (Mortgage {i + 1})",
            value=default_term_1 if i == 0 else default_term_2,
            min_value=1,
            max_value=60,
            step=1,
            key=f"term_{i + 1}_input",
        )

        # Payment frequency is almost always monthly for mortgages in US/Canada
        mortgage_params[f"mortgage_{i + 1}_freq"] = st.selectbox(
            f"Payment Frequency (Mortgage {i + 1})",
            options=[12],
            index=0,
            format_func=lambda x: {12: "Monthly"}[x],
            key=f"freq_{i + 1}_input",
        )  # Fixed to monthly

st.markdown("---")

# --- Generate Schedules and Display ---
if st.button("Generate Amortization Schedules and Compare"):
    st.subheader("Comparison Results")

    # Calculate and display for Mortgage 1
    mortgage1_schedule_df = generate_mortgage_amortization_schedule(
        mortgage_params["mortgage_1_principal"],
        mortgage_params["mortgage_1_rate"],
        mortgage_params["mortgage_1_term"],
        mortgage_params["mortgage_1_freq"],
    )
    mortgage1_monthly_payment_calculated = calculate_mortgage_payment(
        mortgage_params["mortgage_1_principal"],
        mortgage_params["mortgage_1_rate"],
        mortgage_params["mortgage_1_term"],
        mortgage_params["mortgage_1_freq"],
    )

    # Calculate and display for Mortgage 2
    mortgage2_schedule_df = generate_mortgage_amortization_schedule(
        mortgage_params["mortgage_2_principal"],
        mortgage_params["mortgage_2_rate"],
        mortgage_params["mortgage_2_term"],
        mortgage_params["mortgage_2_freq"],
    )
    mortgage2_monthly_payment_calculated = calculate_mortgage_payment(
        mortgage_params["mortgage_2_principal"],
        mortgage_params["mortgage_2_rate"],
        mortgage_params["mortgage_2_term"],
        mortgage_params["mortgage_2_freq"],
    )

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.subheader("Mortgage 1 Summary")
        st.write(f"**Loan Amount:** ${mortgage_params['mortgage_1_principal']:,.2f}")
        st.write(f"**Annual Interest Rate:** {mortgage_params['mortgage_1_rate']:.2%}")
        st.write(f"**Loan Term:** {mortgage_params['mortgage_1_term']} years")
        st.write(
            f"**Calculated Monthly Payment:** ${mortgage1_monthly_payment_calculated:,.2f}"
        )
        st.write(
            f"**Total Payments Over Loan Term:** ${mortgage1_schedule_df['Monthly Payment'].sum():,.2f}"
        )
        st.write(
            f"**Total Principal Paid:** ${mortgage1_schedule_df['Principal Paid'].sum():,.2f}"
        )
        st.write(
            f"**Total Interest Paid:** ${mortgage1_schedule_df['Interest Paid'].sum():,.2f}"
        )

        st.dataframe(
            mortgage1_schedule_df.style.format(
                {
                    "Beginning Balance": "${:,.2f}",
                    "Monthly Payment": "${:,.2f}",
                    "Interest Paid": "${:,.2f}",
                    "Principal Paid": "${:,.2f}",
                    "Ending Balance": "${:,.2f}",
                }
            )
        )

    with col_res2:
        st.subheader("Mortgage 2 Summary")
        st.write(f"**Loan Amount:** ${mortgage_params['mortgage_2_principal']:,.2f}")
        st.write(f"**Annual Interest Rate:** {mortgage_params['mortgage_2_rate']:.2%}")
        st.write(f"**Loan Term:** {mortgage_params['mortgage_2_term']} years")
        st.write(
            f"**Calculated Monthly Payment:** ${mortgage2_monthly_payment_calculated:,.2f}"
        )
        st.write(
            f"**Total Payments Over Loan Term:** ${mortgage2_schedule_df['Monthly Payment'].sum():,.2f}"
        )
        st.write(
            f"**Total Principal Paid:** ${mortgage2_schedule_df['Principal Paid'].sum():,.2f}"
        )
        st.write(
            f"**Total Interest Paid:** ${mortgage2_schedule_df['Interest Paid'].sum():,.2f}"
        )

        st.dataframe(
            mortgage2_schedule_df.style.format(
                {
                    "Beginning Balance": "${:,.2f}",
                    "Monthly Payment": "${:,.2f}",
                    "Interest Paid": "${:,.2f}",
                    "Principal Paid": "${:,.2f}",
                    "Ending Balance": "${:,.2f}",
                }
            )
        )

    st.markdown("---")
    st.subheader("Visual Comparison")

    # Plotting Outstanding Balance over time
    fig_balance, ax_balance = plt.subplots(figsize=(10, 6))
    ax_balance.plot(
        mortgage1_schedule_df["Payment No."],
        mortgage1_schedule_df["Ending Balance"],
        label="Mortgage 1 Outstanding Balance",
        marker="o",
        markersize=3,
        alpha=0.7,
    )
    ax_balance.plot(
        mortgage2_schedule_df["Payment No."],
        mortgage2_schedule_df["Ending Balance"],
        label="Mortgage 2 Outstanding Balance",
        marker="x",
        markersize=3,
        alpha=0.7,
    )

    ax_balance.set_title("Outstanding Loan Balance Over Time", fontsize=14)
    ax_balance.set_xlabel("Payment Number", fontsize=12)
    ax_balance.set_ylabel("Balance ($)", fontsize=12)
    ax_balance.legend(fontsize=10)
    ax_balance.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig_balance)

    # Plotting Total Interest vs. Total Principal (Bar Chart) - This is the requested visualization
    fig_sums, ax_sums = plt.subplots(figsize=(10, 6))
    labels = ["Mortgage 1", "Mortgage 2"]
    total_principal_paid_sum = [
        mortgage1_schedule_df["Principal Paid"].sum(),
        mortgage2_schedule_df["Principal Paid"].sum(),
    ]
    total_interest_paid_sum = [
        mortgage1_schedule_df["Interest Paid"].sum(),
        mortgage2_schedule_df["Interest Paid"].sum(),
    ]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax_sums.bar(
        x - width / 2,
        total_principal_paid_sum,
        width,
        label="Total Principal Paid",
        color="skyblue",
    )
    rects2 = ax_sums.bar(
        x + width / 2,
        total_interest_paid_sum,
        width,
        label="Total Interest Paid",
        color="lightcoral",
    )

    ax_sums.set_ylabel("Amount ($)", fontsize=12)
    ax_sums.set_title(
        "Total Principal vs. Total Interest Paid Over Loan Term", fontsize=14
    )
    ax_sums.set_xticks(x)
    ax_sums.set_xticklabels(labels, fontsize=11)
    ax_sums.legend(fontsize=10)
    ax_sums.grid(axis="y", linestyle="--", alpha=0.7)

    # Function to add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_sums.annotate(
                f"${height:,.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

    st.pyplot(fig_sums)

    st.info("""
    **Understanding the Mortgage Amortization Schedule:**
    * **Loan Amount:** The initial principal borrowed.
    * **Annual Interest Rate:** The stated yearly interest rate of the mortgage.
    * **Loan Term in Years:** The total duration over which the loan is repaid.
    * **Payment Frequency:** How often payments are made (e.g., Monthly = 12 times a year).
    * **Calculated Monthly Payment:** The fixed amount you pay each period, calculated to fully amortize the loan over its term.
    * **Beginning Balance:** The outstanding principal balance at the start of a payment period.
    * **Interest Paid:** The portion of your monthly payment that goes towards paying the interest on the outstanding balance. Calculated as `Beginning Balance * (Annual Interest Rate / Payment Frequency)`.
    * **Principal Paid:** The portion of your monthly payment that reduces the outstanding principal balance. Calculated as `Monthly Payment - Interest Paid`.
    * **Ending Balance:** The remaining principal balance after the payment. Calculated as `Beginning Balance - Principal Paid`. This should reach $0 at the end of the loan term.

    **Key Takeaways from Visualizations:**
    * **Outstanding Loan Balance Over Time:** Illustrates how the principal balance decreases over the loan's life.
    * **Total Principal vs. Total Interest Paid:** This comparison clearly shows the total amount of principal repaid versus the total interest paid over the entire loan term. You'll typically pay significantly more interest in the early years of a mortgage. A shorter loan term often results in substantially less total interest paid, even if the monthly payments are higher.
    """)
