import numpy as np
import pandas as pd
import pickle


products = pd.read_csv('./Data/products.csv')
purchases = pd.read_csv('./Data/purchases.csv')
views = pd.read_csv('./Data/views.csv')

def recommendation_engine(product_list, 
                          CustomerId = None, 
                          purchases = purchases, 
                          views = views, 
                          products = products, 
                          model_file = 'finalized_model.sav',
                         rules_file = "./association_rules.pkl"):
    
    
    """
    Fonksiyon input olarak verilen ürün listesini yeni müşteri ve mevcut müşterilere önermek için sıralar.
    En alakalı ürün ilk sırada olacak şekilde listenin sıralanmış halini döndürür.
    
    Ana yapı 2 kısma ayrılır ve algoritmanın çalışma hiyerarşisi şu şekildedir:
        Yeni müşteriler
            1-En çok satılan ürünler
            2-En çok satılan ürünlere göre eğitilmiş model çıktılarından yüksek satış tahmin edilenler
        Mevcut müşteriler
            1-Müşterinin daha önce satın aldığı ürünler
            2-Daha önce görüntülenmiş ürünler
            3-Daha önce satın alınan ürünler ile birlikte satılan ürünler(Association Rule Mining Model)
            4-En çok satılan ürünler
            5-En çok satılan ürünlere göre eğitilmiş model çıktılarından yüksek satış tahmin edilenler
    
    Parameters:
        product_list (list): Öneri için sıralanması istenen ürün numara listesi
        CustomerId (int): Eğer öneri mevcut müşteriye yapılacaksa ilgili müşteri numarası
        purchases (pandas dataframe): Satın alınan ürünlerin ve müşterilerin ID'lerini içeren dataframe
        views (pandas dataframe): Kişi bazında görüntülenen ürünlerin ve müşterilerin ID'lerini içeren data frame
        products (pandas dataframe): Tüm ürünlerin 7 farklı kırılımda özelliklerini ve ürün ID sini içeren dataframe
        model_file (directory): Hiç satın alınmamış ürünlerin sıralanması için eğitilmiş olan XGBRegressor model dosyası
        rules_file (directory): Bir müşteri tarafından satın alınmış ürünlerin tamamlayıcı ürünlerinin 
                                sıralanması için eğitilmiş olan Association Rule Mining referans tablosu
    
    Returns:
        ordered_list (list): Önemlerine ve önceliklerine göre sıralanmış ürün listesi.
        
    
    """
    
    model = pickle.load(open(model_file, 'rb'))
    
    rules = pd.read_pickle(rules_file)
    
    #ProductID üzerinden grupla, say ve sırala. Referans tablo her ürünün satın alınma frekansını gösteriyor.
    referance_table = purchases[['ProductId', 'IsPurchased']].groupby('ProductId')\
                                                         .count()\
                                                         .sort_values('IsPurchased', ascending = False)\
                                                         .rename({'IsPurchased': 'NofPurchases'}, axis = 1)
    if CustomerId == None:
        #Referans tablosunu input listesine göre filtrele
        product_list1 = list(referance_table[referance_table.index.isin(product_list)].index)

        product_list2  = [product for product in product_list if product not in product_list1 ]
        
        
        order_list = model.predict(products[products.index.isin(product_list2)].drop('ProductId', axis = 1))
        
        
        #Listenin dördüncü aşaması
        similarity_sug = [x for _, x in sorted(zip(order_list, product_list2), reverse =True)]
        
        ordered_list = product_list1 + similarity_sug
        
        
    else:
        CustomerId = int(CustomerId)
        #Müşterinin daha önceden yaptığı alışverişleri filtrele ve aldığı ürünleri sayısına göre azalan şekilde sırala.
        customer_purchases = purchases[purchases.CustomerId == CustomerId].groupby(by= 'ProductId')\
                                                                   .count()\
                                                                   .sort_values('IsPurchased', ascending = False)

        #Daha önceden satın alınan bir ürün var ise o ürünü öner.
        #Listenin sıfırıncı aşaması
        purchase_sug = list(customer_purchases[customer_purchases.index.isin(product_list)].index)

        #Sıralamaya giren ürünleri listeden düşür
        product_list0  = [product for product in product_list if product not in purchase_sug ]
        
        #Daha önceden alınan ürünlerle şuan kontrol ettiğimiz ürünler arasında,
        #association rule mining yaptıgımız referans listeden ilişkili ürünler var ise listeye ekle
        #Listenin birinci aşaması
        
        assoc_sug = list()
        for product in customer_purchases.index:
            sets = list(rules[rules.antecedents.apply(lambda x: x.issubset(frozenset({product})))].consequents)
            dummy_l = [product for product in product_list0 if frozenset({product}) in sets]
            assoc_sug = assoc_sug + dummy_l
        assoc_sug = list(set(assoc_sug))
        
        #Sıralamaya giren ürünleri listeden düşür
        product_list1  = [product for product in product_list0 if product not in assoc_sug ]
        
        #Müşterinin daha önceden yaptığı görüntülemeleri filtrele ve görüntülediği ürünleri sayısına göre azalan şekilde sırala.
        customer_views = views[views.CustomerId == CustomerId].groupby(by= 'ProductId')\
                                                              .count()\
                                                              .sort_values('CustomerId', ascending = False)

        #Daha önceden görüntülenen bir ürün var ise o ürünü öner.
        #Listenin ikinci aşaması
        view_sug = list(customer_views[customer_views.index.isin(product_list1)].index)

        #Sıralamaya giren ürünleri listeden düşür
        product_list2  = [product for product in product_list1 if product not in view_sug ]
        
        #Referans tablosunu input listesine göre filtrele
        #Listenin üçüncü aşaması
        purchase_sug_all = list(referance_table[referance_table.index.isin(product_list2)].index)

        #Sıralamaya giren ürünleri listeden düşür
        product_list3  = [product for product in product_list2 if product not in purchase_sug_all ]
        
        
        # Eğitilmiş olan modeli kullanarak satılma sayısını tahmin et.
        order_list = model.predict(products[products.index.isin(product_list3)].drop('ProductId', axis = 1))

        #Listenin dördüncü aşaması
        similarity_sug = [x for _, x in sorted(zip(order_list, product_list3), reverse =True)]
        
        ordered_list = purchase_sug+assoc_sug+view_sug+purchase_sug_all+similarity_sug
    return ordered_list