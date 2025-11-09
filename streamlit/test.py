prices = [100, 250, 90] 
def discount(lst): 
    return [p*0.9 for p in lst if p>100] 
print(discount(prices))