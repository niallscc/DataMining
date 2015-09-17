#!usr/bin/python
# import scipy as sci
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv(
    "./../data/coupon_list_train_translated.csv",
    header=0,
    usecols=[
        # 'CAPSULE_TEXT',
        # 'GENRE_NAME',
        'PRICE_RATE',
        'CATALOG_PRICE',
        'DISCOUNT_PRICE',
        # 'USABLE_DATE_THU',
        # 'USABLE_DATE_FRI',
        # 'USABLE_DATE_SAT',
        # 'USABLE_DATE_SUN',
        # 'USABLE_DATE_HOLIDAY',
        # 'COUPON_ID_hash'

    ]
)
sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(data)
plt.plot(sklearn_transf)
# plt.plot(sklearn_transf[0:20, 0], sklearn_transf[0:20, 1],
#         'o', markersize=7, color='blue', alpha=0.5, label='class1')
# plt.plot(sklearn_transf[20:40, 0], sklearn_transf[20:40, 1],
#         '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.legend()
plt.title('Transformed samples using sklearn.decomposition.PCA()')
plt.show()
# c_types = []
# for i, row in enumerate(data.values):
#    if row[1] not in c_types:
#        c_types.append(row[1])
