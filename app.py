import streamlit as st
import pandas as pd
import plotly.express as px

# Set the page configuration for a wide layout and a title
st.set_page_config(layout="wide", page_title="Modern Finance Dashboard")

# --- Main Title ---
st.title("Modern Finance Dashboard 📊")

# --- Data Loading and Caching ---
# We use a function with caching to load data efficiently.
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Upload & Filter")
    uploaded_file = st.file_uploader("Upload your Excel file here", type=["xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        user_list = ["Overview"] + sorted(df["UserID"].unique())
        selected_user = st.selectbox("Select a View", user_list)

# --- Main Page Content ---
if not uploaded_file:
    st.info("Please upload your financial data file to get started.")
else:
    # --- OVERVIEW DASHBOARD ---
    if selected_user == "Overview":
        st.header("Dashboard Overview: All Users")
        
        # Create tabs for different sections of the overview
        tab1, tab2, tab3 = st.tabs(["Key Metrics & Distributions", "Profitability Analysis", "Balance Sheet Insights"])

        with tab1:
            st.markdown("### Key Performance Indicators")
            total_revenue = df['Revenue'].sum()
            avg_net_profit_margin = df['NetProfitMargin'].mean()
            total_assets = df['TotalAssets'].sum()
            total_users = df['UserID'].nunique()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
            kpi2.metric("Total Assets", f"${total_assets:,.0f}")
            kpi3.metric("Avg. Net Profit Margin", f"{avg_net_profit_margin:.2f}%")
            kpi4.metric("Total Users", f"{total_users}")
            
            st.markdown("<hr/>", unsafe_allow_html=True)
            
            st.markdown("### Data Distributions")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Revenue Distribution")
                fig_rev_dist = px.histogram(df, x='Revenue', nbins=50, title="Revenue Frequency")
                st.plotly_chart(fig_rev_dist, use_container_width=True)
            with col2:
                st.markdown("#### Top 10 Users by Revenue")
                top_10_users = df.nlargest(10, 'Revenue')
                fig_top_users = px.pie(top_10_users, values='Revenue', names='UserID', title="Revenue Contribution by Top 10 Users")
                st.plotly_chart(fig_top_users, use_container_width=True)

        with tab2:
            st.markdown("### Profitability Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Net Profit Margin Distribution")
                fig_profit_margin = px.box(df, y='NetProfitMargin', title="Spread of Net Profit Margin")
                st.plotly_chart(fig_profit_margin, use_container_width=True)
            with col2:
                st.markdown("#### Net Profit vs. Revenue")
                fig_scatter = px.scatter(df, x='Revenue', y='NetProfit', title="Profitability vs. Scale", labels={'Revenue': 'Total Revenue', 'NetProfit': 'Net Profit'})
                st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.markdown("### Balance Sheet Insights")
            top_20_assets = df.nlargest(20, 'TotalAssets')
            st.markdown("#### Assets vs. Liabilities (Top 20 Users by Assets)")
            fig_assets_liab = px.bar(top_20_assets, x='UserID', y=['TotalAssets', 'TotalLiabilities'], barmode='group', title="Assets vs. Liabilities")
            st.plotly_chart(fig_assets_liab, use_container_width=True)
            
    # --- INDIVIDUAL USER DRILL-DOWN VIEW ---
    else:
        user_data = df[df["UserID"] == selected_user].iloc[0]
        st.header(f"Financial Statement for User: {selected_user}")

        tab1, tab2, tab3 = st.tabs(["Financial Summary", "Margin Analysis", "Asset & Liability Breakdown"])

        with tab1:
            st.subheader("Income Statement Highlights")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Revenue", f"${user_data['Revenue']:,.0f}")
            kpi2.metric("COGS", f"${user_data['COGS']:,.0f}")
            kpi3.metric("Gross Profit", f"${user_data['GrossProfit']:,.0f}")
            kpi4.metric("Net Profit", f"${user_data['NetProfit']:,.0f}")
            
            st.markdown("<hr/>", unsafe_allow_html=True)
            
            st.subheader("Balance Sheet Summary")
            assets_col, liabilities_col = st.columns(2)
            with assets_col:
                st.markdown(f"#### Total Assets: **${user_data['TotalAssets']:,.0f}**")
                st.markdown(f"Cash and Bank: `${user_data['CashAndBankBalance']:,.0f}`")
                st.markdown(f"Accounts Receivable: `${user_data['AccountReceivables']:,.0f}`")
            with liabilities_col:
                st.markdown(f"#### Total Liabilities: **${user_data['TotalLiabilities']:,.0f}**")
                st.markdown(f"Accounts Payable: `${user_data['AccountPayables']:,.0f}`")
                st.markdown(f"Wages Payable: `${user_data['WagesPayable']:,.0f}`")
        
        with tab2:
            st.subheader("Profitability Ratios")
            kpi5, kpi6, kpi7 = st.columns(3)
            kpi5.metric("Gross Profit Margin", f"{user_data['GrossProfitMargin']:.2f}%")
            kpi6.metric("Operating Expense Ratio", f"{user_data['OperatingExpenseRatio']:.2f}%")
            kpi7.metric("Net Profit Margin", f"{user_data['NetProfitMargin']:.2f}%")
            
            # Comparison to average
            avg_npm = df['NetProfitMargin'].mean()
            st.markdown("#### Net Profit Margin vs. Average")
            delta_vs_avg = user_data['NetProfitMargin'] - avg_npm
            st.metric("Company Average Net Profit Margin", f"{avg_npm:.2f}%", f"{delta_vs_avg:.2f}% vs. Average", delta_color="inverse")
            
        with tab3:
            st.subheader("Composition Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Asset Composition")
                asset_data = {
                    'Component': ['Cash/Bank', 'Receivables', 'Inventory', 'Deposits/Advances'],
                    'Value': [user_data['CashAndBankBalance'], user_data['AccountReceivables'], user_data['Inventory'], user_data['DepositsAdvancesPrepayments']]
                }
                asset_df = pd.DataFrame(asset_data)
                fig_asset_donut = px.pie(asset_df, names='Component', values='Value', hole=0.4, title="Assets")
                st.plotly_chart(fig_asset_donut, use_container_width=True)
            with col2:
                st.markdown("#### Liability Composition")
                liability_data = {
                    'Component': ['Payables', 'Wages', 'Provisions', 'Other'],
                    'Value': [user_data['AccountPayables'], user_data['WagesPayable'], user_data['ProvisionsAccruals'], user_data['OtherPayables']]
                }
                liability_df = pd.DataFrame(liability_data)
                fig_liability_donut = px.pie(liability_df, names='Component', values='Value', hole=0.4, title="Liabilities")
                st.plotly_chart(fig_liability_donut, use_container_width=True)

