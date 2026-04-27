import pandas as pd

df       = pd.read_csv('outputs/processed_data.csv')
recs     = pd.read_csv('outputs/recommendations.csv')
all_recs = pd.read_csv('outputs/all_recommendations.csv')
fact     = pd.read_csv('outputs/factory_profile.csv')
prod     = pd.read_csv('outputs/product_profile.csv')

print('=== BASIC STATS ===')
print('Total orders:', len(df))
print('Total products:', df['Product Name'].nunique())
print('Avg lead time:', df['Lead Time'].mean().round(1))
print('Avg profit margin:', df['Profit Margin'].mean().round(1))
print('Total sales:', df['Sales'].sum().round(2))
print('Total gross profit:', df['Gross Profit'].sum().round(2))

print()
print('=== FACTORY PROFILE ===')
print(fact.to_string(index=False))

print()
print('=== TOP RECOMMENDATIONS ===')
cols = ['Product','Current Factory','Alt Factory',
        'LT Reduction %','Profit Impact %','Risk','Confidence']
print(recs[cols].to_string(index=False))

print()
print('=== DIVISION STATS ===')
print(df.groupby('Division')[['Sales','Gross Profit','Cost','Profit Margin']].mean().round(2).to_string())

print()
print('=== REGION LEAD TIME ===')
print(df.groupby('Region')['Lead Time'].mean().round(1).to_string())

print()
print('=== SHIP MODE LEAD TIME ===')
print(df.groupby('Ship Mode')['Lead Time'].mean().round(1).to_string())

print()
print('=== PRODUCT LEAD TIME ===')
print(df.groupby('Product Name')['Lead Time'].mean().round(1).sort_values().to_string())

print()
print('=== TIER COUNTS ===')
def tier(x):
    if x <= 200:   return 'Fast (Tier 1)'
    elif x <= 600: return 'Medium (Tier 2)'
    else:          return 'Slow (Tier 3)'
df['Tier'] = df['Lead Time'].apply(tier)
print(df['Tier'].value_counts().to_string())

print()
print('=== SLOW ROUTES ===')
slow = df.groupby(['Region','Ship Mode'])['Lead Time'].mean().round(1)
print(slow.sort_values(ascending=False).head(8).to_string())