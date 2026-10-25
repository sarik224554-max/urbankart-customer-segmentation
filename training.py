import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#encoding='ISO-8859-1' USE KIYA H QKOKI DATASETS M SPECIAL CHARACTERS HAIN AGR HM DEFAULT UTF-8 SE READ KRENGE TOH 
#UNICODEDECODEERROR AAYEGA

#parse_dates so InvoiceDate is a real date, not text
df=pd.read_csv("data//Online_Retail.csv", encoding='ISO-8859-1', parse_dates=['InvoiceDate'])

#print(df.shape)
#print(df.head(20))
#print(df.columns.tolist())
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#duplicated row 
#print(df.duplicated().sum())


#total cancelled orders
cancelled_orders = df['InvoiceNo'].str.startswith("C").sum()
#print("total cancelled orders: ", cancelled_orders)

# low quantity orders
low_quantity_orders = (df['Quantity'] <= 0).sum()
#print("low quantity orders: ", low_quantity_orders)

#remove cancelled  orders 
df = df[df['Quantity'] > 0]

#low price orders
low_price_orders=(df['UnitPrice']<=0).sum()
#print("low price orders: ", low_price_orders)

#remove low price orders
df=df[df['UnitPrice']>0]

#missing customer ids
missing_customers = df['CustomerID'].isnull().sum()
#print("missing customer ids: ", missing_customers)

#drop rows with missing customer ids 
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)
#print(df.shape)

#duplicated rows
duplicated_rows = df.duplicated().sum()
#print("duplicated rows:", duplicated_rows)

#drop duplicated rows 
df = df.drop_duplicates()
#print(df.shape)

print("after cleaning")
print(df.shape)

print(df["CustomerID"].nunique())

# ===== 3. RFM TABLE BANAO (har customer ki ek row) =====
# KMeans ko har customer ki ek row chahiye, na ki har kharide gaye item ki.
# Recency = last order ke baad kitne din beet chuke hain,
# Frequency = kitne alag invoices/orders kiye,
# Monetary = total kitne paise kharch kiye.

# Fixed reference date use kar rahe hain (aaj ki date nahi),
# taaki code ko alag din run karne par Recency change na ho.
# +1 day add kiya hai taaki sabse recent buyer ki Recency 0 ke bajaye 1 ho.

df['total_price']=df['Quantity']*df['UnitPrice']

snapshot_date=df['InvoiceDate'].max()+pd.Timedelta(days=1)

rfm=df.groupby(['CustomerID']).agg(
    monetary=('total_price','sum'),
    frequency=('InvoiceNo', 'nunique'),
    lastpurchase=('InvoiceDate', 'max')
)

rfm['recency']=(snapshot_date-rfm['lastpurchase']).dt.days

rfm_summary=rfm[['recency','frequency','monetary']].describe().round(1)
print(rfm_summary)
print(rfm.shape)

# ===== 4. TRANSFORM AUR SCALE KARO =====
# Customers ke spending data mein kaafi difference hai
# (median 669 hai, jabki maximum 280,206 hai).
# Kuch customers bahut zyada spend karte hain,
# jiski wajah se KMeans unhi customers ko zyada importance de sakta hai
# aur baaki customers ke actual groups ko sahi se identify nahi kar payega.

rfm_model=rfm[['recency','frequency','monetary']]

#log badhe no. ko daba kr chhota kr deta h

rfm_log=np.log1p(rfm_model)

scaler=StandardScaler()
rfm_scaled=scaler.fit_transform(rfm_log)

#check skew km hua ya nhi 

skew_check=pd.DataFrame({
    "before: " : rfm_model.skew(),
    "after: " : rfm_log.skew(),
}).round(2)

#print(skew_check)

# ===== 5. SEGMENTS KI SANKHYA CHUNO (k) =====
# k = 2 se 8 tak try karo aur inertia (elbow method)
# aur silhouette score ko record karo.
# n_init=10 aur random_state=42 use karne se
# result stable rahega aur code dobara run karne par
# lagbhag wahi result milega.

k_list=[]
for k in range(2,9):
    km=KMeans(n_clusters=k, n_init=10, random_state=42)
    labels=km.fit_predict(rfm_scaled)
    k_list.append({
        "k":k,
        "inertia":km.inertia_,
        "silhouette":silhouette_score(rfm_scaled, labels)
    }) 

k_result=pd.DataFrame(k_list).round(3)
#print(k_result)

#compare by graph

fig, axes=plt.subplots(1, 2, figsize=(10,4) )


axes[0].plot(k_result['k'],k_result['inertia'], marker="o")
axes[0].set_title("elbow: inertia vs k ")
axes[0].set_xlabel("k")

axes[1].plot(k_result['k'],k_result['silhouette'],marker='o',color="green")
axes[1].set_title("silhouette vs k")
axes[1].set_xlabel("k")

plt.tight_layout()
#print(plt.show())

#======== final model==========

km=KMeans(n_clusters=4,n_init=10,random_state=42)
rfm['clusters']=km.fit_predict(rfm_scaled)



profile=rfm.groupby("clusters").agg(
    customers=("recency","size"),
    recency=("recency","median"),
    frequency=("frequency","median"),
    monetary=("monetary","median")
).round(1)

profile['share_%']=(profile["customers"]/profile["customers"].sum()*100).round(1)

segment_names={
    0:"Champions",
    1:"Lost",
    2:"New / Potential Customers",
    3:"Regular"
}

rfm['segment']=rfm['clusters'].map(segment_names)
profile["Segment"] = profile.index.map(segment_names)

print("rich customer",rfm.loc[rfm["monetary"].idxmax()])

print("rfm__",rfm)
print("profile:  ", profile)

revenue=rfm.groupby("segment")['monetary'].sum()
revenue_share=(revenue/revenue.sum()*100).round(1)
revenue_share.sort_values(ascending=False)
print(revenue_share)

share=pd.DataFrame({
    "customers_%" : (rfm['segment'].value_counts(normalize=True)*100).round(1),
    "revenue_%" : revenue_share
}).sort_values("revenue_%",ascending=False)

share.plot(kind="bar",figsize=(10,4),color=["#B4B2A9", "#378ADD"])

plt.title("Champions are 16% of customers but bring 65% of revenue")
plt.ylabel("share (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("segment_share.png", dpi=150, bbox_inches="tight")
plt.show()


segment_summary=rfm.groupby('segment').agg(
    customers=("recency","size"),
    median_days_since_last_order=("recency","median"),
    median_order=("frequency","median"),
    median_spend=("monetary","median"),
    total_revenue=("monetary","sum")
)

segment_summary["customers_%"]=(segment_summary['customers']/segment_summary['customers'].sum()*100).round(1)
segment_summary['revenue_%']=(segment_summary['total_revenue']/segment_summary['total_revenue'].sum()*100).round(1)
segment_summary=segment_summary.sort_values("revenue_%",ascending=False).round(1)

action= {
    "Champions": "Early access to new collection, no discount (Email)",
    "Regular": "Reminder + new products, small incentive after 75+ days gap (Email, WhatsApp opt-in only)",
    "New / Potential Customers": "Free shipping on next order (Email / SMS)",
    "Lost": "One-time limited win-back offer, then stop (Automated email)",
}

segment_summary["marketing_action"]=segment_summary.index.map(action)
print(segment_summary[["customers_%", "revenue_%", "marketing_action"]])

# Customer-level list: this is what the marketing team will actually use

rfm[['recency','frequency','monetary','segment']].to_csv("customers_segments.csv")
segment_summary.to_csv("segment_summary.csv")




import joblib
joblib.dump(km, "model/rfm_kmeans_model.pkl")
joblib.dump(scaler, "model/rfm_scaler.pkl")
joblib.dump(segment_names, "model/segment_names.pkl")