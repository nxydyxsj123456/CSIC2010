from sklearn.linear_model import LogisticRegression
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
param_grid = [
    {
        'C': [0.1, 1, 3, 5, 7],
        'penalty': ['l1', 'l2']
    }
]

#grid_searchgrid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1, cv=5)

LR=LogisticRegression(max_iter=200);

LR.fit(X_train,y_train)

y_pred=LR.predict(X_test_std)


np.save("y_pred2.npy", y_pred);
np.save("y_test2.npy", y_test);


y_pred=np.load("y_pred2.npy");
y_test=np.load("y_test2.npy");


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy=accuracy_score(y_pred,y_test);

precision=precision_score(y_pred,y_test);

recall=recall_score(y_pred,y_test);

f1=f1_score(y_pred,y_test);

print(accuracy,precision,recall,f1);
