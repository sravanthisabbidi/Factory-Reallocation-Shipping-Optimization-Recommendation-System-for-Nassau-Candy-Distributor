import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('charts', exist_ok=True)

# ══════════════════════════════════════════════════════
# STEP 1 — Load & prepare
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

# ══════════════════════════════════════════════════════
# STEP 2 — Create shipping tier (target variable)
# ══════════════════════════════════════════════════════
def assign_tier(lt):
    if lt <= 200:
        return 1   # Fast
    elif lt <= 600:
        return 2   # Medium
    else:
        return 3   # Slow

df['Tier'] = df['Lead Time'].apply(assign_tier)
df['Tier Label'] = df['Tier'].map({1:'Fast (Tier 1)', 2:'Medium (Tier 2)', 3:'Slow (Tier 3)'})

print("=" * 55)
print("TIER DISTRIBUTION")
print(df['Tier Label'].value_counts())

print("\nTIER BY FACTORY")
print(df.groupby(['Factory','Tier Label']).size().unstack(fill_value=0))

print("\nTIER BY PRODUCT")
print(df.groupby(['Product Name','Tier Label']).size().unstack(fill_value=0))

# ══════════════════════════════════════════════════════
# STEP 3 — Feature engineering
# ══════════════════════════════════════════════════════
le_ship    = LabelEncoder()
le_region  = LabelEncoder()
le_factory = LabelEncoder()
le_product = LabelEncoder()
le_div     = LabelEncoder()

df['Ship Mode Enc'] = le_ship.fit_transform(df['Ship Mode'])
df['Region Enc']    = le_region.fit_transform(df['Region'])
df['Factory Enc']   = le_factory.fit_transform(df['Factory'])
df['Product Enc']   = le_product.fit_transform(df['Product Name'])
df['Division Enc']  = le_div.fit_transform(df['Division'])
df['Order Month']   = df['Order Date'].dt.month
df['Order DOW']     = df['Order Date'].dt.dayofweek

# Avg profit margin per factory
fact_margin = df.groupby('Factory')['Profit Margin'].mean()
df['Factory Avg Margin'] = df['Factory'].map(fact_margin)

# Save encoders & lookups
joblib.dump(le_ship,    'models/le_ship.pkl')
joblib.dump(le_region,  'models/le_region.pkl')
joblib.dump(le_factory, 'models/le_factory.pkl')
joblib.dump(le_product, 'models/le_product.pkl')
joblib.dump(le_div,     'models/le_div.pkl')
joblib.dump(fact_margin.to_dict(), 'models/factory_avg_margin.pkl')

FEATURES = [
    'Ship Mode Enc',
    'Region Enc',
    'Factory Enc',
    'Product Enc',
    'Division Enc',
    'Factory Avg Margin',
    'Units',
    'Sales',
    'Cost',
    'Order Month',
    'Order DOW',
]

X = df[FEATURES]
y = df['Tier']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 55)
print("STEP 3 COMPLETE — Train/test split")
print(f"  Training rows : {len(X_train)}")
print(f"  Testing rows  : {len(X_test)}")

# ══════════════════════════════════════════════════════
# STEP 4 — Train 3 models
# ══════════════════════════════════════════════════════
models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n" + "=" * 55)
print("STEP 4 — Training models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'accuracy': accuracy}
    print(f"\n  {name}")
    print(f"    Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")

# ══════════════════════════════════════════════════════
# STEP 5 — Best model details & save
# ══════════════════════════════════════════════════════
best_name  = max(results, key=lambda n: results[n]['accuracy'])
best_model = results[best_name]['model']

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(FEATURES,   'models/features.pkl')
joblib.dump(factory_map,'models/factory_map.pkl')

print("\n" + "=" * 55)
print(f"BEST MODEL : {best_name}")
print(f"  Accuracy : {results[best_name]['accuracy']*100:.1f}%")
print("\nDetailed Report:")
print(classification_report(
    y_test, results[best_name]['preds'],
    target_names=['Fast','Medium','Slow']
))
print("Saved to models/best_model.pkl")

# ══════════════════════════════════════════════════════
# STEP 6 — Save processed dataframe for Streamlit
# ══════════════════════════════════════════════════════
df.to_csv('outputs/processed_data.csv', index=False)
print("Saved: outputs/processed_data.csv")

# ══════════════════════════════════════════════════════
# STEP 7 — Charts
# ══════════════════════════════════════════════════════

# Chart A: Tier distribution
fig, ax = plt.subplots(figsize=(7, 4))
tier_counts = df['Tier Label'].value_counts()
colors = ['#55A868', '#4C72B0', '#C44E52']
bars = ax.bar(tier_counts.index, tier_counts.values, color=colors)
ax.bar_label(bars, padding=3)
ax.set_title('Order Distribution by Shipping Tier')
ax.set_ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('charts/08_tier_distribution.png', dpi=150)
plt.close()
print("Saved: charts/08_tier_distribution.png")

# Chart B: Model accuracy comparison
fig, ax = plt.subplots(figsize=(7, 4))
names = list(results.keys())
accs  = [results[n]['accuracy']*100 for n in names]
bars  = ax.bar(names, accs, color=['#4C72B0','#55A868','#DD8452'])
ax.bar_label(bars, fmt='%.1f%%', padding=3)
ax.set_title('Model Accuracy Comparison')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 110)
ax.set_xticklabels(names, fontsize=9)
plt.tight_layout()
plt.savefig('charts/09_model_accuracy.png', dpi=150)
plt.close()
print("Saved: charts/09_model_accuracy.png")

# Chart C: Feature importance
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(8, 5))
    imp  = best_model.feature_importances_
    idx  = np.argsort(imp)
    bars = ax.barh([FEATURES[i] for i in idx], imp[idx], color='#8172B2')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_title(f'Feature Importance — {best_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('charts/10_feature_importance.png', dpi=150)
    plt.close()
    print("Saved: charts/10_feature_importance.png")

# Chart D: Tier by Factory heatmap
fig, ax = plt.subplots(figsize=(8, 4))
pivot = df.groupby(['Factory','Tier Label']).size().unstack(fill_value=0)
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, str(pivot.values[i,j]),
                ha='center', va='center', fontsize=9, color='black')
ax.set_title('Order Count by Factory and Shipping Tier')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('charts/11_factory_tier_heatmap.png', dpi=150)
plt.close()
print("Saved: charts/11_factory_tier_heatmap.png")

print("\n" + "=" * 55)
print("PHASE 3 & 4 COMPLETE — Models saved!")
print("=" * 55)