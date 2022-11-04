import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

Data = pd.read_excel("Retail.xlsx")
print(Data.head())
print(Data.columns)
print(Data.Country.unique())
Data['Description'] = Data['Description'].str.strip()
Data.dropna(axis = 0, subset = ['InvoiceNo'], inplace = True)
Data['InvoiceNo'] = Data['InvoiceNo'].astype('str')
Data = Data[~Data['InvoiceNo'].str.contains('C')]
basket = (Data[Data['Country'] == "United Kingdom"].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
print(basket)
def Hot_sauce(x):
    if(x <= 0):
        return 0
    if(x>=1):
        return 1
basket_encoded = basket.applymap(Hot_sauce)
basket_UK = basket_encoded
frq_items = apriori(basket_UK, min_support = 0.07, use_colnames = True)
print(frq_items)
rules = association_rules(frq_items, metric = "lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])
print (rules.head())
print (rules [ (rules['lift'] >= 6) & (rules['confidence'] >= 0.8) ])
print(basket['ALARM CLOCK BAKELIKE GREEN'].sum())
print(basket['ALARM CLOCK BAKELIKE RED'].sum())




