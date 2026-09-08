
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Online Retail.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = BASE_DIR / "visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_plot(filename):
    """Save the current figure to the visualizations folder."""
    path = PLOT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def iqr_report(data, column):
    """Calculate IQR limits and number of outliers."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((data[column] < lower) | (data[column] > upper)).sum()

    return {
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": count,
    }


# ============================================================
# 1. LOAD DATA
# ============================================================

section("1. LOAD DATA")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_FILE}\n"
        "Put 'Online Retail.xlsx' in the same folder as preprocessing.py."
    )

df = pd.read_excel(DATA_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
print("\nColumns:")
print(df.columns.tolist())

df.to_csv(OUTPUT_DIR / "raw_data_snapshot.csv", index=False)


# ============================================================
# 2. INITIAL INSPECTION
# ============================================================

section("2. INITIAL INSPECTION")

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nShape:", df.shape)

print("\nDescriptive statistics:")
print(df.describe(include="all"))

df.describe(include="all").to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

section("3. STANDARDIZE COLUMN NAMES")

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

print(df.columns.tolist())


# ============================================================
# 4. MISSING VALUES
# ============================================================

section("4. MISSING VALUE ANALYSIS")

missing = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (df.isna().mean() * 100).round(2)
}).sort_values("Missing_Count", ascending=False)

print(missing)

missing.to_csv(
    OUTPUT_DIR / "missing_value_report.csv"
)


# ============================================================
# 5. DUPLICATES
# ============================================================

section("5. DUPLICATE ANALYSIS")

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count:,}")

if duplicate_count:
    df = df.drop_duplicates().copy()
    print(f"Rows after removing duplicates: {len(df):,}")


# ============================================================
# 6. DATA TYPES
# ============================================================

section("6. DATA TYPE CONVERSION")

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

df["CustomerID"] = pd.to_numeric(
    df["CustomerID"], errors="coerce"
).astype("Int64")

print(df.dtypes)


# ============================================================
# 7. CANCELLATIONS AND RETURNS
# ============================================================

section("7. CANCELLATION / RETURN ANALYSIS")

df["IsCancelled"] = (
    df["InvoiceNo"]
    .astype(str)
    .str.upper()
    .str.startswith("C")
)

cancelled_df = df[df["IsCancelled"]].copy()
returns_df = df[df["Quantity"] < 0].copy()

print(f"Cancelled rows: {len(cancelled_df):,}")
print(f"Negative-quantity rows: {len(returns_df):,}")

cancelled_df.to_csv(
    OUTPUT_DIR / "cancelled_transactions.csv",
    index=False
)

returns_df.to_csv(
    OUTPUT_DIR / "return_transactions.csv",
    index=False
)


# ============================================================
# 8. INVALID VALUES
# ============================================================

section("8. INVALID / UNUSUAL VALUES")

invalid_price = df[df["UnitPrice"] <= 0]
zero_quantity = df[df["Quantity"] == 0]

print(f"UnitPrice <= 0: {len(invalid_price):,}")
print(f"Quantity == 0: {len(zero_quantity):,}")

invalid_df = df[
    (df["UnitPrice"] <= 0) |
    (df["Quantity"] == 0)
].copy()

invalid_df.to_csv(
    OUTPUT_DIR / "invalid_transactions.csv",
    index=False
)


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

section("9. FEATURE ENGINEERING")

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Month_Name"] = df["InvoiceDate"].dt.strftime("%b")
df["Day"] = df["InvoiceDate"].dt.day
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
df["Hour"] = df["InvoiceDate"].dt.hour
df["InvoiceDay"] = df["InvoiceDate"].dt.date

print(
    df[
        [
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "Revenue",
            "Year",
            "Month",
            "DayOfWeek",
            "Hour",
        ]
    ].head()
)


# ============================================================
# 10. CLEAN SALES DATASET
# ============================================================

section("10. CLEAN SALES DATASET")

sales_df = df[
    (~df["IsCancelled"]) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

print(f"Original rows: {len(df):,}")
print(f"Clean sales rows: {len(sales_df):,}")

sales_df.to_csv(
    OUTPUT_DIR / "online_retail_cleaned_sales.csv",
    index=False
)


# ============================================================
# 11. BUSINESS METRICS
# ============================================================

section("11. BUSINESS METRICS")

metrics = {
    "Total Revenue (£)": sales_df["Revenue"].sum(),
    "Total Units Sold": sales_df["Quantity"].sum(),
    "Unique Products": sales_df["StockCode"].nunique(),
    "Unique Customers": sales_df["CustomerID"].nunique(),
    "Unique Countries": sales_df["Country"].nunique(),
    "Unique Invoices": sales_df["InvoiceNo"].nunique(),
}

for key, value in metrics.items():
    if "£" in key:
        print(f"{key}: £{value:,.2f}")
    else:
        print(f"{key}: {value:,.0f}")

pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Value": list(metrics.values())
}).to_csv(
    OUTPUT_DIR / "business_metrics.csv",
    index=False
)


# ============================================================
# 12. MONTHLY REVENUE
# ============================================================

section("12. MONTHLY REVENUE")

monthly_revenue = (
    sales_df
    .set_index("InvoiceDate")
    .resample("MS")["Revenue"]
    .sum()
)

monthly_revenue_df = monthly_revenue.reset_index()
monthly_revenue_df.columns = ["Month", "Revenue"]

print(monthly_revenue_df)

monthly_revenue_df.to_csv(
    OUTPUT_DIR / "monthly_revenue.csv",
    index=False
)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_revenue_df["Month"],
    monthly_revenue_df["Revenue"],
    marker="o"
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.25)
save_plot("01_monthly_revenue.png")


# ============================================================
# 13. MONTHLY ORDERS
# ============================================================

section("13. MONTHLY ORDERS")

monthly_orders = (
    sales_df
    .groupby(
        sales_df["InvoiceDate"].dt.to_period("M")
    )["InvoiceNo"]
    .nunique()
)

monthly_orders_df = monthly_orders.reset_index()
monthly_orders_df.columns = ["Month", "Orders"]
monthly_orders_df["Month"] = monthly_orders_df["Month"].astype(str)

print(monthly_orders_df)

monthly_orders_df.to_csv(
    OUTPUT_DIR / "monthly_orders.csv",
    index=False
)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_orders_df["Month"],
    monthly_orders_df["Orders"],
    marker="o"
)
plt.title("Monthly Number of Orders")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.25)
save_plot("02_monthly_orders.png")


# ============================================================
# 14. DAY-OF-WEEK ANALYSIS
# ============================================================

section("14. DAY-OF-WEEK ANALYSIS")

days = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_revenue = (
    sales_df
    .groupby("DayOfWeek")["Revenue"]
    .sum()
    .reindex(days)
)

print(weekday_revenue)

weekday_revenue.to_csv(
    OUTPUT_DIR / "weekday_revenue.csv"
)

plt.figure(figsize=(10, 5))
plt.bar(
    weekday_revenue.index,
    weekday_revenue.values
)
plt.title("Revenue by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.25)
save_plot("03_weekday_revenue.png")


# ============================================================
# 15. HOURLY ANALYSIS
# ============================================================

section("15. HOURLY ORDER ANALYSIS")

hourly_orders = (
    sales_df
    .groupby("Hour")["InvoiceNo"]
    .nunique()
)

hourly_orders.to_csv(
    OUTPUT_DIR / "hourly_orders.csv"
)

plt.figure(figsize=(10, 5))
plt.plot(
    hourly_orders.index,
    hourly_orders.values,
    marker="o"
)
plt.title("Orders by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Orders")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.25)
save_plot("04_hourly_orders.png")


# ============================================================
# 16. TOP PRODUCTS BY REVENUE
# ============================================================

section("16. TOP PRODUCTS BY REVENUE")

top_products_revenue = (
    sales_df
    .groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products_revenue)

top_products_revenue.to_csv(
    OUTPUT_DIR / "top_products_by_revenue.csv"
)

plt.figure(figsize=(10, 6))
top_products_revenue.sort_values().plot(
    kind="barh"
)
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Product")
save_plot("05_top_products_by_revenue.png")


# ============================================================
# 17. TOP PRODUCTS BY QUANTITY
# ============================================================

section("17. TOP PRODUCTS BY QUANTITY")

top_products_quantity = (
    sales_df
    .groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products_quantity)

top_products_quantity.to_csv(
    OUTPUT_DIR / "top_products_by_quantity.csv"
)

plt.figure(figsize=(10, 6))
top_products_quantity.sort_values().plot(
    kind="barh"
)
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
save_plot("06_top_products_by_quantity.png")


# ============================================================
# 18. COUNTRY ANALYSIS
# ============================================================

section("18. COUNTRY-WISE REVENUE")

country_revenue = (
    sales_df
    .groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print(country_revenue.head(10))

country_revenue.to_csv(
    OUTPUT_DIR / "country_revenue.csv"
)

plt.figure(figsize=(10, 6))
country_revenue.head(10).sort_values().plot(
    kind="barh"
)
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Country")
save_plot("07_country_revenue.png")


# ============================================================
# 19. CUSTOMER ANALYSIS
# ============================================================

section("19. CUSTOMER ANALYSIS")

customer_df = sales_df.dropna(
    subset=["CustomerID"]
).copy()

customer_revenue = (
    customer_df
    .groupby("CustomerID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

customer_orders = (
    customer_df
    .groupby("CustomerID")["InvoiceNo"]
    .nunique()
)

customer_summary = pd.DataFrame({
    "Revenue": customer_revenue,
    "Orders": customer_orders
})

customer_summary["Average_Order_Value"] = (
    customer_summary["Revenue"] /
    customer_summary["Orders"]
)

customer_summary = customer_summary.sort_values(
    "Revenue",
    ascending=False
)

customer_summary.to_csv(
    OUTPUT_DIR / "customer_summary.csv"
)

print("\nTop 10 customers:")
print(customer_summary.head(10))


# Customer revenue distribution
plt.figure(figsize=(10, 6))
plt.hist(
    customer_summary["Revenue"],
    bins=50,
    edgecolor="black"
)
plt.title("Distribution of Customer Revenue")
plt.xlabel("Customer Revenue (£)")
plt.ylabel("Number of Customers")
save_plot("08_customer_revenue_distribution.png")


# Top customers
plt.figure(figsize=(10, 6))
customer_summary.head(10)["Revenue"].sort_values().plot(
    kind="barh"
)
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Customer ID")
save_plot("09_top_customers.png")


# ============================================================
# 20. CORRELATION
# ============================================================

section("20. CORRELATION ANALYSIS")

numeric_cols = [
    "Quantity",
    "UnitPrice",
    "Revenue"
]

corr = sales_df[numeric_cols].corr()

print(corr)

corr.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(7, 5))
plt.imshow(
    corr,
    interpolation="nearest",
    aspect="auto"
)
plt.colorbar(label="Correlation")
plt.xticks(range(3), numeric_cols)
plt.yticks(range(3), numeric_cols)
plt.title("Correlation Matrix")

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

save_plot("10_correlation_heatmap.png")


# ============================================================
# 21. QUANTITY VS UNIT PRICE
# ============================================================

section("21. QUANTITY VS UNIT PRICE")

sample = sales_df[
    ["Quantity", "UnitPrice"]
].sample(
    n=min(10000, len(sales_df)),
    random_state=42
)

plt.figure(figsize=(10, 6))
plt.scatter(
    sample["Quantity"],
    sample["UnitPrice"],
    alpha=0.35,
    s=12
)
plt.title("Quantity vs Unit Price")
plt.xlabel("Quantity")
plt.ylabel("Unit Price (£)")
plt.grid(alpha=0.2)
save_plot("11_quantity_vs_unit_price.png")


# ============================================================
# 22. OUTLIER DETECTION
# ============================================================

section("22. OUTLIER DETECTION USING IQR")

outlier_results = []

for column in ["Quantity", "UnitPrice", "Revenue"]:
    result = iqr_report(sales_df, column)
    outlier_results.append(result)

outlier_df = pd.DataFrame(outlier_results)

print(outlier_df)

outlier_df.to_csv(
    OUTPUT_DIR / "outlier_report.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.boxplot(
    [
        sales_df["Quantity"],
        sales_df["UnitPrice"],
        sales_df["Revenue"]
    ],
    labels=["Quantity", "Unit Price", "Revenue"],
    showfliers=True
)
plt.title("Boxplots for Outlier Detection")
plt.ylabel("Value")
save_plot("12_outlier_boxplots.png")


# ============================================================
# 23. CANCELLATION TREND
# ============================================================

section("23. MONTHLY CANCELLATION TREND")

if not cancelled_df.empty:
    cancellation_monthly = (
        cancelled_df
        .set_index("InvoiceDate")
        .resample("MS")
        .size()
    )

    cancellation_monthly_df = (
        cancellation_monthly
        .reset_index(name="Cancelled_Rows")
    )

    cancellation_monthly_df.to_csv(
        OUTPUT_DIR / "monthly_cancellations.csv",
        index=False
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        cancellation_monthly_df["InvoiceDate"],
        cancellation_monthly_df["Cancelled_Rows"],
        marker="o"
    )
    plt.title("Monthly Cancellation Trend")
    plt.xlabel("Month")
    plt.ylabel("Cancelled Transaction Rows")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)
    save_plot("13_monthly_cancellations.png")


# ============================================================
# 24. UK VS INTERNATIONAL
# ============================================================

section("24. UK VS INTERNATIONAL REVENUE")

sales_df["Market"] = np.where(
    sales_df["Country"].eq("United Kingdom"),
    "United Kingdom",
    "International"
)

market_revenue = (
    sales_df
    .groupby("Market")["Revenue"]
    .sum()
)

print(market_revenue)

market_revenue.to_csv(
    OUTPUT_DIR / "uk_vs_international_revenue.csv"
)

plt.figure(figsize=(7, 5))
plt.pie(
    market_revenue.values,
    labels=market_revenue.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("UK vs International Revenue Share")
save_plot("14_uk_vs_international_revenue.png")


# ============================================================
# 25. FINISH
# ============================================================

section("EDA PROCESSING COMPLETE")

print(f"""
Input file:
    {DATA_FILE}

Cleaned data:
    {OUTPUT_DIR / "online_retail_cleaned_sales.csv"}

Summary files:
    {OUTPUT_DIR}

Charts:
    {PLOT_DIR}

Total revenue:
    £{sales_df["Revenue"].sum():,.2f}

Total units:
    {sales_df["Quantity"].sum():,.0f}

Products:
    {sales_df["StockCode"].nunique():,}

Customers:
    {sales_df["CustomerID"].nunique():,}

Countries:
    {sales_df["Country"].nunique():,}

Invoices:
    {sales_df["InvoiceNo"].nunique():,}
""")
