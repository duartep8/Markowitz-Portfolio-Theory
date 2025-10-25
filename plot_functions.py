import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.ticker as mtick

from datetime import datetime

def prepare_portfolio_data():
    """
    Prepares the initial portfolio data from the provided list.
    In a real-world scenario, you might load this from a CSV.
    Returns:
        pd.DataFrame: The prepared portfolio data.
    """
    # Data from the table, structured for DataFrame creation
    data = {
        'Company': [
            'Synopsys, Inc.', 'ASML Holding N.V.', 'SAP SE', 'Infineon Technologies AG', 
            'Siemens Aktiengesellschaft', 'Schneider Electric SE', 'Thales Group', 'Airbus SE',
            'Allianz SE', 'BNP Paribas SA', 'ING Groep N.V.',
            'TotalEnergies SE', 'Iberdrola SA', 'Enel SpA', 'EDP : Energias de Portugal SA',
            'LVMH Moët Hennessy - Louis Vuitton', 'Industria de Diseño Textil SA', 'Ferrari N.V.', 
            'Koninklijke Ahold Delhaize N.V.',
            'Sanofi', 'Siemens Healthineers AG',
            'BASF SE', 'Linde plc',
            'Deutsche Telekom AG', 'Telefónica SA'
        ],
        'Ticker': [
            'SYP.DE', 'ASML.AS', 'SAP.DE', 'IFX.DE',
            'SIE.DE', 'SU.PA', 'HO.PA', 'AIR.PA',
            'ALV.DE', 'BNP.PA', 'INGA.AS',
            'TTE.PA', 'IBE.MC', 'ENEL.MI', 'EDP.LS',
            'MC.PA', 'ITX.MC', 'RACE.MI', 'AD.AS',
            'SAN.PA', 'SHL.DE',
            'BAS.DE', 'LIN.DE',
            'DTE.DE', 'TEF.MC'
        ],
        'Market_Cap_B': [
            74.9, 347.2, 271.8, 43.5,
            190.8, 142.8, 53.6, 164.1,
            136.0, 76.8, 60.2,
            117.0, 114.5, 85.9, 18.4,
            308.3, 154.9, 62.9, 33.0,
            108.1, 55.6,
            39.0, 180.6,
            142.1, 25.7
        ],
        'Credit_Rating': [
            'BBB', 'No info', 'A+', 'BBB+',
            'AA-', 'A', 'A-', 'A',
            'AA', 'A+', 'A-',
            'A+', 'BBB+', 'BBB', 'BBB',
            'AA-', 'No info', 'No info', 'No info',
            'AA', 'No Info',
            'A-', 'A',
            'BBB+', 'BBB-'
        ],
        'Sector': [
            'Technology', 'Technology', 'Technology', 'Technology',
            'Industrials', 'Industrials', 'Industrials', 'Industrials',
            'Financial Services', 'Financial Services', 'Financial Services',
            'Energy', 'Utilities', 'Utilities', 'Utilities',
            'Consumer Cyclical', 'Consumer Cyclical', 'Consumer Cyclical', 'Consumer Defensive',
            'Healthcare', 'Healthcare',
            'Basic Materials', 'Basic Materials',
            'Communication Services', 'Communication Services'
        ],
        'Industry': [
            'Software - Infrastructure', 'Semiconductor Equipment & Materials', 'Software - Application', 'Semiconductors',
            'Specialty Industrial Machinery', 'Specialty Industrial Machinery', 'Aerospace & Defense', 'Aerospace & Defense',
            'Insurance - Diversified', 'Banks - Regional', 'Banks - Diversified',
            'Oil & Gas Integrated', 'Utilities - Diversified', 'Utilities - Diversified', 'Utilities - Diversified',
            'Luxury Goods', 'Apparel Retail', 'Auto Manufacturers', 'Grocery Stores',
            'Drug Manufacturers - General', 'Medical Devices',
            'Chemicals', 'Chemicals',
            'Telecom Services', 'Telecom Services'
        ],
        'Country': [
            'USA', 'Netherlands', 'Germany', 'Germany',
            'Germany', 'France', 'France', 'Netherlands',
            'Germany', 'Luxembourg', 'Netherlands',
            'France', 'Spain', 'Italy', 'Portugal',
            'France', 'Spain', 'Italy', 'Netherlands',
            'France', 'Germany',
            'Germany', 'Ireland',
            'Germany', 'Spain'
        ]
    }
    
    portfolio_df = pd.DataFrame(data)
    return portfolio_df

def plot_portfolio_weights(weights, tickers, file_path=None, title="Optimal Portfolio Weights"):
    
    weights_pct = np.array(weights) * 100
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))

    positive_color = '#005A9C'  # Blue for positive
    negative_color = '#D50000'  # Red for negative
    
    # Create the bars
    bars = ax.bar(tickers, weights_pct, 
                  color=[positive_color if w >= 0 else negative_color for w in weights_pct], 
                  edgecolor='black', linewidth=0.7)

    # --- START: Updated Labeling Loop ---
    # This loop correctly places labels just outside the bars
    # and uses vertical alignment (va) to keep them neat.
    for bar, w in zip(bars, weights_pct):
        if w >= 0:
            # For positive bars, place label slightly above
            label_y_pos = w + 0.2  
            va = 'bottom' # Align the bottom of the text with label_y_pos
        else:
            # For negative bars, place label slightly below
            label_y_pos = w - 0.2  
            va = 'top'    # Align the top of the text with label_y_pos
        
        ax.text(bar.get_x() + bar.get_width() / 2, label_y_pos, f"{w:.2f}%", 
                ha='center', va=va, fontsize=11, fontweight='bold', 
                color='black', fontfamily='serif')
    # --- END: Updated Labeling Loop ---

    ax.axhline(0, color='black', linewidth=1.2)
    
    # --- ADDED: Set Y-Axis Limits ---
    # This creates padding so the labels for the tallest/shortest bars
    # (e.g., 10.0% and -8.0%) are not cut off by the chart edge.
    # We add a 2.5% buffer to the min and max weights.
    ax.set_ylim(weights_pct.min() - 2.5, weights_pct.max() + 2.5)
    # --- END: Added Line ---

    # Set titles and labels
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, fontfamily='serif')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=14, fontfamily='serif')
    ax.set_xlabel('Stock Ticker', fontsize=14, fontfamily='serif')
    
    # Format ticks and spines
    plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='serif')
    plt.yticks(fontsize=12, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)  # Keep gridlines behind bars
    
    plt.tight_layout()

    # Save or show the plot
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved portfolio weights plot to: {file_path}")
        plt.close(fig)
    else:
        # Check if the backend is interactive before trying to show the plot
        backend = matplotlib.get_backend()
        if 'agg' in backend.lower():
            print(f"⚠️ Warning: Matplotlib is using the non-interactive '{backend}' backend. Plot cannot be shown.")
            print("To view the plot, provide a 'file_path' argument to save it to a file, or switch to an interactive backend.")
            plt.close(fig)
        else:
            plt.show()

"""
def create_portfolio_treemap(df: pd.DataFrame, output_filename: str = 'portfolio_composition_final_with_totals.png'):
    print("Generating corrected executive-level visualization...")

    # --- 1. CALCULATE TOTALS ---
    total_market_cap = df['Market_Cap_B'].sum()
    
    # --- THIS IS THE NEW, CRUCIAL STEP ---
    # We calculate industry totals first using groupby.
    industry_totals = df.groupby('Industry')['Market_Cap_B'].sum()
    # Now, we create a new column with formatted labels that include these totals.
    df['Industry_Label'] = df['Industry'].map(
        lambda name: f"{name}<br><b>Total: {industry_totals[name]:,.2f}B</b>"
    )
    # --- END OF NEW STEP ---

    title_text = "<b>Portfolio Composition by Industry and Market Capitalization</b>"
    subtitle_text = f"Total Portfolio Market Cap: {total_market_cap:,.2f}B €"

    fig = px.treemap(
        df,
        # We use the new 'Industry_Label' column for the path.
        path=[px.Constant("Total Portfolio"), 'Industry_Label', 'Ticker'],
        values='Market_Cap_B',
        color='Market_Cap_B',
        color_continuous_scale='Blues',
        hover_data={'Company': True, 'Market_Cap_B': ':.2f'},
        branchvalues='total'
    )

    # --- 3. APPLY PROFESSIONAL LAYOUT AND STYLING ---
    fig.update_layout(
        title_text=f"{title_text}<br><sup>{subtitle_text}</sup>",
        title_font=dict(size=24, family="Arial, sans-serif"),
        font=dict(family="Arial, sans-serif", size=14, color="#333"),
        margin=dict(t=80, l=20, r=20, b=80),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        coloraxis_showscale=False,
        annotations=[
            dict(
                text=f"Data as of: {datetime.now().strftime('%Y-%m-%d')}. All values in EUR billions.",
                align='left', showarrow=False, xref='paper', yref='paper',
                x=0, y=-0.08
            )
        ]
    )

    # --- 4. ENHANCE TRACES AND TOOLTIPS ---
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:,.2f}B",
        textposition='middle center',
        insidetextfont=dict(size=14),
        marker=dict(cornerradius=5),
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Company: %{customdata[0]}<br>" +
            "Market Cap: %{value:,.2f}B<br>" +
            "Share of Total Portfolio: %{percentRoot:.2%}<extra></extra>"
        )
    )

    pio.write_image(fig, output_filename, width=1600, height=1000, scale=3)
    print(f"Successfully saved the corrected plot as '{output_filename}'")


def plot_headquarters_map(portfolio_df: pd.DataFrame, 
                          output_filename: str = 'portfolio_headquarters_map.png', 
                          title: str = 'Portfolio Headquarters Distribution (by Number of Companies)'):
    
    print("Aggregating data for headquarters map...")
    
    # --- 1. Aggregate Data by Country ---
    country_agg = portfolio_df.groupby('Country').agg(
        Total_Market_Cap_B=('Market_Cap_B', 'sum'),
        Company_Count=('Ticker', 'count')
    ).reset_index()

    # --- 2. Define Coordinates for Text Labels ---
    # Updated with Luxembourg
    country_coords = {
        'France': {'lat': 46.2276, 'lon': 2.2137},
        'USA': {'lat': 38.9637, 'lon': -95.7129},
        'Netherlands': {'lat': 52.1326, 'lon': 5.2913},
        'Germany': {'lat': 51.1657, 'lon': 10.4515},
        'Spain': {'lat': 40.4637, 'lon': -3.7492},
        'Italy': {'lat': 41.8719, 'lon': 12.5674},
        'Portugal': {'lat': 38.7369, 'lon': -9.1427},
        'Ireland': {'lat': 53.4129, 'lon': -8.2439},
        'Luxembourg': {'lat': 49.8153, 'lon': 6.1296}
    }
    
    # Map coordinates to the aggregated DataFrame
    country_agg['lat'] = country_agg['Country'].map(lambda x: country_coords.get(x, {}).get('lat'))
    country_agg['lon'] = country_agg['Country'].map(lambda x: country_coords.get(x, {}).get('lon'))

    print(country_agg)

    # --- 3. Create the Base Choropleth Map (colored by count) ---
    fig = px.choropleth(
        country_agg,
        locations="Country",
        locationmode="country names",
        color="Company_Count",
        hover_name="Country",
        hover_data={
            'Country': False,
            'Company_Count': True,
            'Total_Market_Cap_B': ':.2fB'
        },
        color_continuous_scale=px.colors.sequential.Blues,
        title=title,
    )
    
    # --- 4. Add Text Labels using a Scattergeo Trace ---
    fig.add_trace(go.Scattergeo(
        locations=country_agg['Country'],
        locationmode="country names",
        lat=country_agg['lat'],
        lon=country_agg['lon'],
        text=country_agg['Company_Count'],
        mode="text",
        textfont=dict(
            family="Arial, sans-serif",
            size=14,
            color="black",
        ),
        showlegend=False
    ))
    
    # --- 5. Refine Layout ---
    fig.update_layout(
        margin={"r":0, "t":50, "l":0, "b":0},
        font=dict(family="Arial, sans-serif", size=14),
        title_font=dict(size=20),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='mercator'
        ),
        coloraxis_colorbar=dict(
            title="Number of Companies"
        )
    )
    
    # --- 6. Save Plot to PNG ---
    try:
        pio.write_image(fig, output_filename, scale=2, width=1200, height=800)
        print(f"✅ Saved static map image with text to: {output_filename}")
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        print("Please ensure you have 'kaleido' installed (pip install kaleido)")
"""

"""
def create_portfolio_sunburst(df: pd.DataFrame, output_filename: str = 'portfolio_composition_sunburst.png'):

    print("Generating circular sunburst visualization...")

    # CALCULATE TOTAL 
    total_market_cap = df['Market_Cap_B'].sum()
    
    # DEFINE TITLES 
    title_text = "<b>Portfolio Composition by Industry and Market Capitalization</b>"
    subtitle_text = f"Total Portfolio Market Cap: {total_market_cap:,.2f}B €"

    # CREATE THE SUNBURST CHART 
    fig = px.sunburst(
        df,
        path=[px.Constant("Total Portfolio"), 'Industry', 'Ticker'],
        values='Market_Cap_B',
        color='Industry', 
        color_discrete_sequence=px.colors.qualitative.Pastel,
        branchvalues='total'
    )

    # APPLY PROFESSIONAL LAYOUT AND STYLING
    fig.update_layout(
        title_text=f"{title_text}<br><sup>{subtitle_text}</sup>",
        title_font=dict(size=24, family="Arial, sans-serif"),
        font=dict(family="Arial, sans-serif", size=14, color="#333"),
        margin=dict(t=100, l=40, r=40, b=100), 
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        annotations=[
            dict(
                text=f"Data as of: {datetime.now().strftime('%Y-%m-%d')}. All values in EUR billions.",
                align='left', showarrow=False, xref='paper', yref='paper',
                x=0.05, y=-0.08
            )
        ]
    )

    # ENHANCE TRACES AND TOOLTIPS
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:,.2f}B<br>%{percentRoot:.2%}",
        insidetextfont=dict(size=12),
        marker_line=dict(color='white', width=2), 
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Market Cap: %{value:,.2f}B<br>" +
            "Share of Parent (e.g., Industry): %{percentParent:.2%}<br>" +
            "Share of Total Portfolio: %{percentRoot:.2%}<extra></extra>"
        )
    )

    # SAVE THE IMAGE
    pio.write_image(fig, output_filename, width=1200, height=1200, scale=3) 
    print(f"Successfully saved the sunburst plot as '{output_filename}'")
"""

def plot_market_cap_by_sector(df, filename='market_cap_by_sector.png'):
    """
    Plots market cap distribution by sector as a bar chart with percentages.
    
    Args:
        df: DataFrame with 'Sector' and 'Market_Cap_B' columns
        filename: Output filename for the PNG file
    """
    # Group by sector and sum market cap
    sector_data = df.groupby('Sector')['Market_Cap_B'].sum().sort_values(ascending=False)
    
    # Calculate percentages
    total = sector_data.sum()
    percentages = (sector_data / total) * 100
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Create the bars
    bars = ax.bar(sector_data.index, percentages, color='#005A9C', 
                   edgecolor='black', linewidth=0.7)
    
    # Add labels above bars
    for bar, pct in zip(bars, percentages):
        label_y_pos = pct + 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, label_y_pos, f"{pct:.2f}%",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='black', fontfamily='serif')
    
    ax.axhline(0, color='black', linewidth=1.2)
    
    # Set Y-Axis Limits with padding
    ax.set_ylim(0, percentages.max() + 5)
    
    # Set titles and labels
    ax.set_title('Market Cap Distribution by Sector', fontsize=18, fontweight='bold', 
                 pad=20, fontfamily='serif')
    ax.set_ylabel('Percentage of Total Market Cap (%)', fontsize=14, fontfamily='serif')
    ax.set_xlabel('Sector', fontsize=14, fontfamily='serif')
    
    # Format ticks and spines
    plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='serif')
    plt.yticks(fontsize=12, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved sector market cap plot to: {filename}")
    plt.close(fig)


def plot_market_cap_by_industry(df, filename='market_cap_by_industry.png'):
    """
    Generate a professional market capitalization distribution visualization by industry.
    
    Creates a bar chart displaying the percentage allocation of market cap
    across industries, formatted for institutional fund reporting.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Industry' and 'Market_Cap_B' columns
    filename : str, optional
        Output path for PNG file (default: 'market_cap_by_industry.png')
    
    Returns
    -------
    None
        Saves the visualization to the specified filename
    """
    # Aggregate market cap by industry
    industry_data = df.groupby('Industry')['Market_Cap_B'].sum().sort_values(ascending=False)
    
    # Calculate percentage allocation
    total_market_cap = industry_data.sum()
    allocation_pct = (industry_data / total_market_cap) * 100
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Create the bars
    bars = ax.bar(industry_data.index, allocation_pct, color='#005A9C',
                   edgecolor='black', linewidth=0.7)
    
    # Add labels above bars
    for bar, pct in zip(bars, allocation_pct):
        label_y_pos = pct + 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, label_y_pos, f"{pct:.2f}%",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='black', fontfamily='serif')
    
    ax.axhline(0, color='black', linewidth=1.2)
    
    # Set Y-Axis Limits with padding
    ax.set_ylim(0, allocation_pct.max() + 5)
    
    # Set titles and labels
    ax.set_title('Market Cap Distribution by Industry', fontsize=18, fontweight='bold',
                 pad=20, fontfamily='serif')
    ax.set_ylabel('Percentage of Total Market Cap (%)', fontsize=14, fontfamily='serif')
    ax.set_xlabel('Industry', fontsize=14, fontfamily='serif')
    
    # Format ticks and spines
    plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='serif')
    plt.yticks(fontsize=12, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved industry market cap plot to: {filename}")
    print(f"Total sectors analyzed: {len(industry_data)}")
    print(f"Total market cap: €{total_market_cap:.2f}B")
    plt.close(fig)


def plot_industry_weights(optimal_weights, tickers, portfolio_df, output_filename='industry_weights.png'):
    """
    Plot portfolio weight distribution by industry.
    
    Args:
        optimal_weights: Array of optimized weights
        tickers: List of ticker symbols
        portfolio_df: DataFrame with 'Ticker' and 'Industry' columns
        output_filename: Output filename for the PNG file
    """
    print("Aggregating weights by industry and plotting...")

    # Create a DataFrame from the optimized weights and tickers
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': optimal_weights
    })

    # Merge with the portfolio info to get the industry for each ticker
    merged_df = pd.merge(
        weights_df,
        portfolio_df[['Ticker', 'Industry']],
        on='Ticker',
        how='left'
    )

    # Aggregate (sum) the weights by industry
    industry_weights = merged_df.groupby('Industry')['Weight'].sum()
    
    # Sort for better visualization
    industry_weights = industry_weights.sort_values(ascending=False)
    
    # Convert to percentage
    weights_pct = industry_weights * 100

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))
    
    positive_color = '#005A9C'  # Blue for positive
    negative_color = '#D50000'  # Red for negative
    
    # Create the bars with conditional coloring
    bars = ax.bar(weights_pct.index, weights_pct.values,
                   color=[positive_color if w >= 0 else negative_color for w in weights_pct.values],
                   edgecolor='black', linewidth=0.7)
    
    # Add labels outside bars with proper positioning
    for bar, w in zip(bars, weights_pct.values):
        if w >= 0:
            # For positive bars, place label slightly above
            label_y_pos = w + 0.5
            va = 'bottom'
        else:
            # For negative bars, place label slightly below
            label_y_pos = w - 0.5
            va = 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, label_y_pos, f"{w:.2f}%",
                ha='center', va=va, fontsize=11, fontweight='bold',
                color='black', fontfamily='serif')
    
    ax.axhline(0, color='black', linewidth=1.2)
    
    # Set Y-Axis Limits with padding for both positive and negative values
    ax.set_ylim(weights_pct.min() - 2.5, weights_pct.max() + 2.5)
    
    # Set titles and labels
    ax.set_title('Portfolio Weight by Industry', fontsize=18, fontweight='bold',
                 pad=20, fontfamily='serif')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=14, fontfamily='serif')
    ax.set_xlabel('Industry', fontsize=14, fontfamily='serif')
    
    # Format ticks and spines
    plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='serif')
    plt.yticks(fontsize=12, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved industry weight plot to: {output_filename}")
    plt.close(fig)

def plot_companies_by_country(df, filename='companies_by_country.png'):
    """
    Plots the number of companies by country as a bar chart.
    
    Args:
        df: DataFrame with 'Country' column
        filename: Output filename for the PNG file
    """
    # Count companies by country
    country_counts = df['Country'].value_counts().sort_values(ascending=False)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Create the bars
    bars = ax.bar(country_counts.index, country_counts.values, color='#005A9C',
                   edgecolor='black', linewidth=0.7)
    
    # Add labels above bars
    for bar, count in zip(bars, country_counts.values):
        label_y_pos = count + 0.15
        ax.text(bar.get_x() + bar.get_width() / 2, label_y_pos, f"{count}",
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='black', fontfamily='serif')
    
    ax.axhline(0, color='black', linewidth=1.2)
    
    # Set Y-Axis Limits with padding
    ax.set_ylim(0, country_counts.max() + 1.5)
    
    # Set titles and labels
    ax.set_title('Number of Companies by Country', fontsize=18, fontweight='bold',
                 pad=20, fontfamily='serif')
    ax.set_ylabel('Number of Companies', fontsize=14, fontfamily='serif')
    ax.set_xlabel('Country', fontsize=14, fontfamily='serif')
    
    # Format ticks and spines
    plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='serif')
    plt.yticks(fontsize=12, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved companies by country plot to: {filename}")
    print(f"Total countries: {len(country_counts)}")
    print(f"Total companies: {country_counts.sum()}")
    plt.close(fig)

def plot_sector_weights(optimal_weights, tickers, portfolio_df, output_filename='sector_weights.png'):
    """
    Plot portfolio weight distribution by sector.
    
    Args:
        optimal_weights: Array of optimized weights
        tickers: List of ticker symbols
        portfolio_df: DataFrame with 'Ticker' and 'Sector' columns
        output_filename: Output filename for the PNG file
    """
    print("Aggregating weights by sector and plotting...")

    # Create a DataFrame from the optimized weights and tickers
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': optimal_weights
    })

    # Merge with the portfolio info to get the sector for each ticker
    merged_df = pd.merge(
        weights_df,
        portfolio_df[['Ticker', 'Sector']],
        on='Ticker',
        how='left'
    )

    # Aggregate (sum) the weights by sector
    sector_weights = merged_df.groupby('Sector')['Weight'].sum()
    
    # Sort for better visualization
    sector_weights = sector_weights.sort_values(ascending=False)
    
    # Convert to percentage
    weights_pct = sector_weights * 100

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))
    
    positive_color = '#005A9C'  # Blue for positive
    negative_color = '#D50000'  # Red for negative
    
    # Create the bars with conditional coloring
    bars = ax.bar(weights_pct.index, weights_pct.values,
                   color=[positive_color if w >= 0 else negative_color for w in weights_pct.values],
                   edgecolor='black', linewidth=0.7)
    
    # Add labels outside bars with proper positioning
    for bar, w in zip(bars, weights_pct.values):
        if w >= 0:
            # For positive bars, place label slightly above
            label_y_pos = w + 0.5
            va = 'bottom'
        else:
            # For negative bars, place label slightly below
            label_y_pos = w - 0.5
            va = 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, label_y_pos, f"{w:.2f}%",
                ha='center', va=va, fontsize=11, fontweight='bold',
                color='black', fontfamily='serif')
    
    ax.axhline(0, color='black', linewidth=1.2)
    
    # Set Y-Axis Limits with padding for both positive and negative values
    ax.set_ylim(weights_pct.min() - 2.5, weights_pct.max() + 2.5)
    
    # Set titles and labels
    ax.set_title('Portfolio Weight by Sector', fontsize=18, fontweight='bold',
                 pad=20, fontfamily='serif')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=14, fontfamily='serif')
    ax.set_xlabel('Sector', fontsize=14, fontfamily='serif')
    
    # Format ticks and spines
    plt.xticks(rotation=45, ha='right', fontsize=12, fontfamily='serif')
    plt.yticks(fontsize=12, fontfamily='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved sector weight plot to: {output_filename}")
    plt.close(fig)


# --- Main Execution Block ---
def main():

    print("Starting portfolio analysis...")
    portfolio_data = prepare_portfolio_data()
    
    # Call the visualization function to create and save the treemap
    plot_market_cap_by_sector(portfolio_data)
    plot_market_cap_by_industry(portfolio_data)
    plot_companies_by_country(portfolio_data)

    # plot_headquarters_map(portfolio_data)
    # create_portfolio_treemap(portfolio_data)
    # create_portfolio_sunburst(portfolio_data)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
