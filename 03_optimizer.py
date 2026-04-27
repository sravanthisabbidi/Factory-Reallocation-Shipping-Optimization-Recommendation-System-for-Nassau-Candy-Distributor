import pandas as pd
import numpy as np
import joblib
import os

os.makedirs('outputs', exist_ok=True)

# ══════════════════════════════════════════════════════
# STEP 1 — Load & prepare data
# ══════════════════════════════════════════════════════
df = pd.read_csv("Nassau Candy Distributor.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
df['Ship Date']  = df['Ship Date'] - pd.DateOffset(days=730)
df['Lead Time']  = (df['Ship Date'] - df['Order Date']).dt.days

factory_map = {
    'Wonka Bar - Nutty Crunch Surprise'   : "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows'           : "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious'      : "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate'          : "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel'   : "Wicked Choccy's",
    'Laffy Taffy'                         : 'Sugar Shack',
    'SweeTARTS'                           : 'Sugar Shack',
    'Nerds'                               : 'Sugar Shack',
    'Fun Dip'                             : 'Sugar Shack',
    'Fizzy Lifting Drinks'                : 'Sugar Shack',
    'Everlasting Gobstopper'              : 'Secret Factory',
    'Lickable Wallpaper'                  : 'Secret Factory',
    'Wonka Gum'                           : 'Secret Factory',
    'Kazookles'                           : 'The Other Factory',
    'Hair Toffee'                         : 'The Other Factory',
}
df['Factory'] = df['Product Name'].map(factory_map)
df['Profit Margin'] = (df['Gross Profit'] / df['Sales'] * 100).round(2)

ALL_FACTORIES = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    'Sugar Shack',
    'Secret Factory',
    'The Other Factory',
]

# ══════════════════════════════════════════════════════
# STEP 2 — Build factory performance profiles
# ══════════════════════════════════════════════════════
factory_profile = df.groupby('Factory').agg(
    Avg_Lead_Time   = ('Lead Time',    'mean'),
    Avg_Profit_Margin = ('Profit Margin','mean'),
    Avg_Sales       = ('Sales',        'mean'),
    Avg_Gross_Profit= ('Gross Profit', 'mean'),
    Total_Orders    = ('Row ID',       'count'),
).round(2)

print("=" * 60)
print("FACTORY PERFORMANCE PROFILES")
print(factory_profile.to_string())

# ══════════════════════════════════════════════════════
# STEP 3 — Build product performance profiles
# ══════════════════════════════════════════════════════
product_profile = df.groupby(['Product Name','Factory']).agg(
    Avg_Lead_Time    = ('Lead Time',     'mean'),
    Avg_Profit_Margin= ('Profit Margin', 'mean'),
    Avg_Sales        = ('Sales',         'mean'),
    Total_Orders     = ('Row ID',        'count'),
).round(2).reset_index()

print("\n" + "=" * 60)
print("PRODUCT PERFORMANCE PROFILES (current factory)")
print(product_profile[['Product Name','Factory',
                        'Avg_Lead_Time','Avg_Profit_Margin',
                        'Total_Orders']].to_string(index=False))

# ══════════════════════════════════════════════════════
# STEP 4 — Simulate reassignment to all factories
# ══════════════════════════════════════════════════════
# For each product, estimate performance at every factory
# using factory-level averages as the expected baseline

print("\n" + "=" * 60)
print("STEP 4 — Running factory reassignment simulation...")

records = []
for product in df['Product Name'].unique():
    prod_df  = df[df['Product Name'] == product]
    cur_fact = factory_map.get(product, 'Unknown')
    cur_lt   = prod_df['Lead Time'].mean()
    cur_pm   = prod_df['Profit Margin'].mean()
    cur_orders = len(prod_df)

    for alt_factory in ALL_FACTORIES:
        # Estimate lead time at alt factory using factory avg
        alt_lt = factory_profile.loc[alt_factory, 'Avg_Lead_Time']
        alt_pm = factory_profile.loc[alt_factory, 'Avg_Profit_Margin']

        lt_reduction     = cur_lt - alt_lt
        lt_reduction_pct = (lt_reduction / cur_lt * 100)
        profit_impact    = alt_pm - cur_pm

        # Confidence: higher if factory has more orders
        factory_orders   = factory_profile.loc[alt_factory, 'Total_Orders']
        confidence       = min(95, round(50 + (factory_orders / 200), 1))

        # Risk: high if profit margin drops more than 10%
        if profit_impact < -10:
            risk = 'High'
        elif profit_impact < 0:
            risk = 'Medium'
        else:
            risk = 'Low'

        records.append({
            'Product'           : product,
            'Current Factory'   : cur_fact,
            'Alt Factory'       : alt_factory,
            'Is Current'        : cur_fact == alt_factory,
            'Current LT'        : round(cur_lt, 1),
            'Alt LT'            : round(alt_lt, 1),
            'LT Reduction'      : round(lt_reduction, 1),
            'LT Reduction %'    : round(lt_reduction_pct, 1),
            'Current Margin %'  : round(cur_pm, 1),
            'Alt Margin %'      : round(alt_pm, 1),
            'Profit Impact %'   : round(profit_impact, 1),
            'Confidence'        : confidence,
            'Risk'              : risk,
            'Current Orders'    : cur_orders,
        })

sim_df = pd.DataFrame(records)

# ══════════════════════════════════════════════════════
# STEP 5 — Generate recommendations
# ══════════════════════════════════════════════════════
# Only keep alternatives (not current factory)
# Only keep if lead time improves
# Rank by lead time reduction

recs = sim_df[
    (~sim_df['Is Current']) &
    (sim_df['LT Reduction'] > 0)
].copy()

# Score: weight speed vs profit (60% speed, 40% profit)
recs['Score'] = (
    0.6 * recs['LT Reduction %'] +
    0.4 * recs['Profit Impact %']
)

recs = recs.sort_values('Score', ascending=False)

print("\n" + "=" * 60)
print("TOP FACTORY REASSIGNMENT RECOMMENDATIONS")
print("=" * 60)

top_recs = recs.groupby('Product').first().reset_index()
top_recs = top_recs.sort_values('LT Reduction %', ascending=False)

for i, row in top_recs.iterrows():
    print(f"\nProduct  : {row['Product']}")
    print(f"  From   : {row['Current Factory']}  →  To: {row['Alt Factory']}")
    print(f"  Lead Time : {row['Current LT']} → {row['Alt LT']} days  "
          f"({row['LT Reduction %']:+.1f}%)")
    print(f"  Margin    : {row['Current Margin %']}% → {row['Alt Margin %']}%  "
          f"(impact: {row['Profit Impact %']:+.1f}%)")
    print(f"  Risk      : {row['Risk']}   Confidence: {row['Confidence']}%")

# ══════════════════════════════════════════════════════
# STEP 6 — Save outputs
# ══════════════════════════════════════════════════════
sim_df.to_csv('outputs/simulation_results.csv', index=False)
top_recs.to_csv('outputs/recommendations.csv', index=False)
recs.to_csv('outputs/all_recommendations.csv', index=False)

# Save factory profile for Streamlit
factory_profile.reset_index().to_csv(
    'outputs/factory_profile.csv', index=False)
product_profile.to_csv(
    'outputs/product_profile.csv', index=False)

# Save processed df
df.to_csv('outputs/processed_data.csv', index=False)

print("\n" + "=" * 60)
print("FILES SAVED TO outputs/ FOLDER:")
print("  simulation_results.csv")
print("  recommendations.csv")
print("  all_recommendations.csv")
print("  factory_profile.csv")
print("  product_profile.csv")
print("  processed_data.csv")
print("\nPHASE 5 COMPLETE — Optimization engine done!")
print("=" * 60)