"""AI Inventory & Demand Planning Assistant: Streamlit dashboard.

Run it from the project folder with:
    streamlit run src/dashboard.py

Streamlit re-runs this whole script from top to bottom every time you interact with the page
(pick a dropdown value, tick a checkbox, ...). That's why data loading is cached, and why the page is
built by calling one function per section in order at the bottom of the file.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Build file paths from this file's location, so the dashboard works no matter which folder
# you start it from. Path(__file__) is this file; .parent goes up one folder (src/ -> project root).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

URGENCY_ORDER = ["High", "Medium", "Low"]

# Light background tints for the urgency column, with dark text so they stay readable
URGENCY_STYLES = {
    "High": "background-color: #f8d0cc; color: #0b0b0b",    # red-ish
    "Medium": "background-color: #fbe7a8; color: #0b0b0b",  # yellow-ish
    "Low": "background-color: #cdeccf; color: #0b0b0b",     # green-ish
}

ACTUAL_COLOR = "#2a78d6"    # blue: real historical sales
FORECAST_COLOR = "#eb6834"  # orange: forecast
HISTORY_DAYS = 90


# --- 1. Page config -----------------------------------------------------------------------------
# st.set_page_config() sets page-level options: the browser tab title and the layout.
# layout="wide" uses the full browser width instead of a narrow centred column, which suits tables.
# It must be the first Streamlit command in the script.
st.set_page_config(page_title="AI Inventory & Demand Planning Assistant", layout="wide")


# --- 2. Data loading ----------------------------------------------------------------------------
# @st.cache_data is a "decorator": it wraps the function below and remembers its result.
# The first time the function runs, Streamlit stores the returned table. On every later re-run
# (every click on the page), it hands back the stored copy instead of reading the CSV again.
# Loading 913,000 rows once instead of on every click keeps the dashboard fast.
# The cache refreshes automatically if you edit the function's code, or when you choose
# "Clear cache" from the menu (top right, the ⋮ button) after a CSV file has changed.
@st.cache_data
def load_sales():
    return pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])


@st.cache_data
def load_forecast():
    # The day-by-day 14-day forecast, saved by notebooks/inventory_logic.ipynb.
    # Reading the saved file is instant; re-running the model here would take about 40 seconds.
    return pd.read_csv(DATA_DIR / "demand_forecast_14day.csv", parse_dates=["date"])


@st.cache_data
def load_inventory_status():
    return pd.read_csv(DATA_DIR / "inventory_status.csv")


@st.cache_data
def load_recommendations():
    recommendations_df = pd.read_csv(DATA_DIR / "ai_recommendations.csv")

    # Sort most urgent first. A categorical column tells pandas that High < Medium < Low,
    # rather than sorting the words alphabetically (High, Low, Medium).
    recommendations_df["urgency"] = pd.Categorical(
        recommendations_df["urgency"], categories=URGENCY_ORDER, ordered=True
    )
    return recommendations_df.sort_values(["urgency", "days_of_stock_left"]).reset_index(drop=True)


# --- 7. Sidebar ---------------------------------------------------------------------------------
def render_sidebar():
    # st.sidebar is a panel on the left of the page. Anything called on it (st.sidebar.markdown,
    # or inside "with st.sidebar:") appears there instead of in the main area.
    with st.sidebar:
        st.header("About this dashboard")
        st.markdown(
            "This dashboard brings together the stages of the project:\n\n"
            "1. **Demand forecasting**: an XGBoost model trained on 5 years of daily sales "
            "forecasts the next 14 days for all 500 store-item combinations.\n"
            "2. **Inventory logic**: forecasts are turned into stock-out risk, overstock flags "
            "and reorder quantities.\n"
            "3. **AI recommendations**: Google Gemini turns each flagged product's numbers into a "
            "plain-English purchasing recommendation."
        )
        # st.warning() shows a highlighted yellow box, so this caveat can't be missed
        st.warning(
            "**Current stock levels are simulated.** The source dataset contains sales only, with no "
            "real inventory data. Each product's current stock is a random 5 to 20 days' worth of its "
            "recent sales, for demonstration. Stock-out risk, overstock flags, reorder quantities and "
            "AI recommendations all depend on these simulated numbers."
        )


# --- 3. Summary metrics -------------------------------------------------------------------------
def render_summary_metrics(inventory_df):
    # st.columns(3) splits the row into 3 side-by-side columns; "with col:" puts content in one
    col_total, col_stockout, col_overstock = st.columns(3)

    # st.metric() shows a large number with a small label above it: a "metric card"
    with col_total:
        st.metric("Products tracked", len(inventory_df))
    with col_stockout:
        st.metric("Stock-out risk", int(inventory_df["stockout_risk"].sum()))
    with col_overstock:
        st.metric("Overstocked", int(inventory_df["overstock_flag"].sum()))


# --- 4. AI recommendations ----------------------------------------------------------------------
def highlight_urgency(value):
    """Return the CSS style for one urgency cell (empty string = no styling)."""
    return URGENCY_STYLES.get(value, "")


def render_recommendations(recommendations_df, inventory_df):
    # st.header() writes a large section title
    st.header("AI Purchasing Recommendations")

    # Compare against the number of flagged products, so this works with 5 rows, 72 rows, or anything else
    num_flagged = int((inventory_df["stockout_risk"] | inventory_df["overstock_flag"]).sum())
    num_done = len(recommendations_df)
    if num_done < num_flagged:
        # st.info() shows a blue information box
        st.info(f"Recommendations are still being generated ({num_done} of {num_flagged} complete).")

    # .style.map() applies highlight_urgency to every cell in the urgency column; st.dataframe
    # displays those colours. .format() controls how the decimal columns are shown.
    styled_table = (
        recommendations_df.style
        .map(highlight_urgency, subset=["urgency"])
        .format({"days_of_stock_left": "{:.1f}", "avg_daily_predicted_demand": "{:.1f}"})
    )

    # st.dataframe() shows an interactive table: sortable columns, scrolling, search.
    # hide_index=True hides pandas' row numbers. column_config sets nicer headers, and a wider
    # column for the reason text.
    st.dataframe(
        styled_table,
        hide_index=True,
        width="stretch",
        column_config={
            "store": "Store",
            "item": "Item",
            "action": "Action",
            "urgency": "Urgency",
            "reason": st.column_config.TextColumn("Reason", width="large"),
            "days_of_stock_left": "Days of stock left",
            "reorder_qty": "Reorder qty",
            "current_stock": "Current stock",
            "avg_daily_predicted_demand": "Forecast / day",
        },
    )


# --- 5. Forecast explorer -----------------------------------------------------------------------
def build_sales_chart(product_sales, product_forecast):
    """Line chart of the product's recent actual sales, followed by its 14-day daily forecast."""
    figure = go.Figure()

    # go.Scatter with mode="lines" draws a line chart
    figure.add_trace(go.Scatter(
        x=product_sales["date"],
        y=product_sales["sales"],
        mode="lines",
        name="Actual daily sales",
        line={"color": ACTUAL_COLOR, "width": 2},
        hovertemplate="%{x|%a %d %b %Y}<br>%{y} units<extra></extra>",
    ))

    # The forecast line starts AT the last actual data point, so the two lines join up with no gap:
    # "here's what happened" flows straight into "here's what's predicted next".
    # pd.concat() stacks tables on top of each other: the last actual day, then the 14 forecast days.
    last_actual = product_sales.tail(1).rename(columns={"sales": "predicted_sales"})
    forecast_line = pd.concat([last_actual[["date", "predicted_sales"]],
                               product_forecast[["date", "predicted_sales"]]])

    # customdata passes an extra per-point label to the tooltip: the joining point is a real value
    point_labels = ["Actual"] + ["Forecast"] * len(product_forecast)
    figure.add_trace(go.Scatter(
        x=forecast_line["date"],
        y=forecast_line["predicted_sales"],
        mode="lines",
        name="Forecast (next 14 days)",
        line={"color": FORECAST_COLOR, "width": 2, "dash": "dash"},
        customdata=point_labels,
        hovertemplate="%{x|%a %d %b %Y}<br>%{customdata}: %{y:.1f} units<extra></extra>",
    ))

    # A faint vertical line marking where the real data ends and the forecast begins
    figure.add_vline(x=product_sales["date"].max(), line={"color": "#9a9993", "width": 1, "dash": "dot"})

    figure.update_layout(
        yaxis_title="Units sold per day",
        hovermode="x unified",  # one tooltip showing every line at the hovered date
        legend={"orientation": "h", "y": 1.1},
        margin={"t": 40, "b": 20},
        height=400,
    )
    figure.update_yaxes(rangemode="tozero")  # start the y-axis at 0 so the scale isn't misleading
    return figure


def render_forecast_explorer(sales_df, forecast_df, inventory_df):
    st.header("Forecast Explorer")

    # Two dropdowns side by side. The options come from the data rather than being typed in.
    col_store, col_item = st.columns(2)
    with col_store:
        # st.selectbox() is a dropdown; it returns whichever option is currently selected
        store = st.selectbox("Store", sorted(inventory_df["store"].unique()))
    with col_item:
        item = st.selectbox("Item", sorted(inventory_df["item"].unique()))

    # This product's last 90 days of real sales
    product_sales = sales_df[(sales_df["store"] == store) & (sales_df["item"] == item)]
    cutoff_date = product_sales["date"].max() - pd.Timedelta(days=HISTORY_DAYS - 1)
    product_sales = product_sales[product_sales["date"] >= cutoff_date].sort_values("date")

    # This product's row in the inventory table (.iloc[0] takes the first and only matching row)
    product_status = inventory_df[(inventory_df["store"] == store) & (inventory_df["item"] == item)].iloc[0]

    # This product's 14 forecast days, in date order
    product_forecast = forecast_df[(forecast_df["store"] == store) & (forecast_df["item"] == item)].sort_values("date")

    st.subheader(f"Store {store}, item {item}: last {HISTORY_DAYS} days of sales and 14-day forecast")
    # st.plotly_chart() displays an interactive Plotly chart (hover, zoom, pan)
    st.plotly_chart(build_sales_chart(product_sales, product_forecast), width="stretch")

    render_product_cards(product_status)


def get_product_status(product_status):
    """Decide a product's status. Returns (text, colour, icon) for the Status card."""
    if product_status["overstock_flag"]:
        return "Overstocked", "violet", ":material/inventory_2:"
    if product_status["stockout_risk"]:
        # Stock runs out before a new delivery could arrive: the most serious case
        return "Reorder needed — stock-out risk", "red", ":material/error:"
    if product_status["reorder_qty"] > 0:
        # Stock lasts past the delivery time, but not with the safety buffer on top
        return "Reorder needed", "orange", ":material/warning:"
    return "OK — no reorder needed", "green", ":material/check_circle:"


def colored_card(label, value, color, icon):
    """A card like st.metric(), but with the value in colour plus an icon.

    st.metric() can't colour its main number, so we build the card ourselves:
    - st.container(border=True) draws a box with a border, matching st.metric(border=True)
    - Streamlit markdown understands :color[text] for coloured text and :material/name: for icons.
      The icon and words carry the meaning too, so it's still clear for colour-blind readers.
    """
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"#### :{color}[{icon} {value}]")


def render_product_cards(product_status):
    """The selected product's inventory numbers, with its status and reorder quantity highlighted."""
    status_text, status_color, status_icon = get_product_status(product_status)
    reorder_qty = int(product_status["reorder_qty"])

    col_stock, col_days, col_reorder, col_status = st.columns(4)
    with col_stock:
        # border=True gives the metric a box, so it lines up with the coloured cards
        st.metric("Current stock (simulated)", f"{product_status['current_stock']:,} units", border=True)
    with col_days:
        st.metric("Days of stock left", f"{product_status['days_of_stock_left']:.1f}", border=True)
    with col_reorder:
        if reorder_qty > 0:
            # Make the number stand out, in the same colour as the status
            colored_card("Reorder quantity", f"Order {reorder_qty:,} units", status_color, ":material/shopping_cart:")
        else:
            # 0 is the healthy case here: say so, instead of showing a bare "0"
            colored_card("Reorder quantity", "0 units (none needed)", "gray", ":material/remove:")
    with col_status:
        colored_card("Status", status_text, status_color, status_icon)


# --- 6. Full inventory status -------------------------------------------------------------------
def render_full_inventory(inventory_df):
    st.header("Full Inventory Status")

    # Filters in one row above the table
    col_stores, col_stockout, col_overstock = st.columns([3, 1, 1])
    with col_stores:
        # st.multiselect() lets you pick several options; it returns the list of selected ones.
        # An empty selection means "all stores".
        selected_stores = st.multiselect(
            "Stores", sorted(inventory_df["store"].unique()), placeholder="All stores"
        )
    with col_stockout:
        # st.checkbox() returns True when ticked, False otherwise
        only_stockout = st.checkbox("Show only stock-out risk")
    with col_overstock:
        only_overstock = st.checkbox("Show only overstocked")

    filtered_df = inventory_df
    if selected_stores:
        filtered_df = filtered_df[filtered_df["store"].isin(selected_stores)]
    if only_stockout:
        filtered_df = filtered_df[filtered_df["stockout_risk"]]
    if only_overstock:
        filtered_df = filtered_df[filtered_df["overstock_flag"]]

    # st.caption() writes small grey text
    st.caption(f"Showing {len(filtered_df)} of {len(inventory_df)} products")
    st.dataframe(
        filtered_df,
        hide_index=True,
        width="stretch",
        column_config={
            "store": "Store",
            "item": "Item",
            "current_stock": "Current stock (simulated)",
            "avg_daily_predicted_demand": st.column_config.NumberColumn("Forecast / day", format="%.1f"),
            "days_of_stock_left": st.column_config.NumberColumn("Days of stock left", format="%.1f"),
            "stockout_risk": "Stock-out risk",
            "reorder_qty": "Reorder qty",
            "overstock_flag": "Overstocked",
        },
    )


# --- Page layout --------------------------------------------------------------------------------
def main():
    sales_df = load_sales()
    forecast_df = load_forecast()
    inventory_df = load_inventory_status()
    recommendations_df = load_recommendations()

    # st.title() writes the big heading at the top of the page
    st.title("AI Inventory & Demand Planning Assistant")
    render_sidebar()
    render_summary_metrics(inventory_df)

    # st.divider() draws a horizontal line between sections
    st.divider()
    render_recommendations(recommendations_df, inventory_df)
    st.divider()
    render_forecast_explorer(sales_df, forecast_df, inventory_df)
    st.divider()
    render_full_inventory(inventory_df)


main()
