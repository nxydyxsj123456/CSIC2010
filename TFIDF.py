from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if len(d) > 0:
            result.append(d)
    return result

normal_requests = load_data('normal.txt')
anomalous_requests = load_data('anomalous.txt')

all_requests = normal_requests + anomalous_requests
y_normal = [0] * len(normal_requests)
y_anomalous = [1] * len(anomalous_requests)
y = y_normal + y_anomalous


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=0.0, analyzer="word", sublinear_tf=True)
X = vectorizer.fit_transform(all_requests)

print(len(vectorizer.vocabulary_.keys())) #总共 33550个种类 统计每个种类的 TF(某个词占文章总次数，出现次数多说明重要) IDF  （总文章数/ 有该词的文章+1，每个文章都有这个词说明他没有区分能力）


print(X.todense())
X=np.array(X.todense());

print(X.shape)

# 划分测试集和训练集
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)


from sklearn.preprocessing import StandardScaler

# 数据归一化
standardScalar = StandardScaler()
standardScalar.fit(X_train)
X_train = standardScalar.transform(X_train)
X_test_std = standardScalar.transform(X_test)


# 网格搜索的参数
# param_grid = [
#     {
#         'weights': ['uniform'],
#         'n_neighbors': [i for i in range(2, 11)] #从1开始容易过拟合
#     },
#     {
#         'weights': ['distance'],
#         'n_neighbors': [i for i in range(2, 11)],
#         'p': [i for i in range(1, 6)]
#     }
# ]

# cv其实也是一个超参数，一般越大越好，但是越大训练时间越长
# grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, cv=5)
# grid_search.fit(X_train,y_train)

KNClassifier=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')

KNClassifier.fit(X_train,y_train)

y_pred=KNClassifier.predict(X_test_std)

print(accuracy_score(y_pred,y_test));


