# <--Thêm thư viện-->
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
# <--Thêm thư viện end-->

# <--Thêm dữ liệu-->
data = 'adult.csv'
df = pd.read_csv(data, header=None, sep=',\s')
# <--Thêm dữ liệu end-->

# <--Thăm dò dữ liệu-->
print(df.shape)  # Lệnh shape trả về kích thước của mảng hoặc ma trận,tức là số hàng và số cột.
print(df.head())  # head() là một phương thức giúp bạn xem 5 hàng đầu tiên của DataFrame đó
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns = col_names

print(df.columns)
print(df.head())
print(df.info())  # info() của DataFrame để hiển thị thông tin chi tiết về DataFrame
categorical = [var for var in df.columns if df[var].dtype == 'O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)
print(df[categorical].head())
print(df[categorical].isnull().sum())
for var in categorical:
    print(df[var].value_counts())  # value_counts():số lần xuất hiện
for var in categorical:
    print(df[var].value_counts() / float(len(df)))
df.workclass.unique()  # +unique() thường được sử dụng để trích xuất các giá trị duy nhất từ một chuỗi dữ liệu
df.workclass.value_counts()
df['workclass'].replace('?', np.NaN, inplace=True)
df.workclass.value_counts()
df.occupation.unique()
df.occupation.value_counts()
df['occupation'].replace('?', np.NaN, inplace=True)
df.occupation.value_counts()
df.native_country.unique()
df.native_country.value_counts()
df['native_country'].replace('?', np.NaN, inplace=True)
df.native_country.value_counts()
df[categorical].isnull().sum()
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')
numerical = [var for var in df.columns if df[var].dtype != 'O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
df[numerical].head()
df[numerical].isnull().sum()
# # <--Thăm dò dữ end -->
#
#
#
#
#
# <--Khai báo vector đặc trưng và biến mục tiêu -->
X = df.drop(['income'], axis=1)  # xóa cột income ở tập dữ liệu đi (axis = 1 -> xóa cột,axis = 0 ->xóa hàng ),tức X là DataFrame sau khi đã xóa cột income
y = df['income']  # y chứa cột income từ tập dữ liệu ban đầu
# <--Khai báo vector đặc trưng và biến mục tiêu end-->
#
# <--Chia dữ liệu thành tập huấn luyện và kiểm tra riêng biệt -->
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# train_test_split:được sử dụng để chia dữ liệu thành hai phần: một phần dùng để huấn luyện mô hình và phần còn lại dùng để kiểm tra hiệu suất mô hình
# test_size=0.3: Chia dữ liệu sao cho 30% của nó được sử dụng cho kiểm tra (X_test và y_test), và 70% được sử dụng cho huấn luyện (X_train và y_train).
# random_state=0: Thiết lập một seed (hạt giống) để đảm bảo kết quả có thể được tái tạo nếu cần thiết.
X_train.shape, X_test.shape
# <--Chia dữ liệu thành tập huấn luyện và kiểm tra riêng biệt end-->
#
#
#
#
#
# <--Kỹ thuật tính năng -->
print(X_train.dtypes)
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
print(categorical)

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
print(numerical)

X_train[categorical].isnull().mean()  # mean() là một phương thức được sử dụng để tính giá trị trung bình của một dãy số hoặc một mảng
for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print(col, (X_train[col].isnull().mean()))
# điền giá trị thiếu (null) trong các cột 'workclass', 'occupation', và 'native_country' của cả X_train và X_test bằng giá trị mode (chế độ, giá trị xuất hiện nhiều nhất) của từng cột trong X_train
for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0],
                            inplace=True)  # fillna là một phương thức trong pandas được sử dụng để điền (thay thế) các giá trị thiếu
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)
print(X_train[categorical].isnull().sum())
print(X_test[categorical].isnull().sum())
print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(categorical)
X_train[categorical].head()
#
# thư viện mã
import category_encoders as ce
# bước này phục vụ tiền xử lý dữ liệu   - One-Hot Encoding: Converts categorical variables into a numerical format.
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship',
                                 'race', 'sex', 'native_country'])  # mã hóa

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.head())
print(X_train.shape)
print(X_test.head())
print(X_test.shape)
# <--Kỹ thuật tính năng end-->
#
#
#
# <--Chia tỷ lệ tính năng -->
cols = X_train.columns  #Robust Scaling: Standardizes the numerical features to ensure that they are on similar scales and are less sensitive to outliers.
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols]) #Conversion Back to DataFrames: Maintains the original column names and facilitates further analysis and interpretation.
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.head())
# <--Chia tỷ lệ tính năng end-->
#
#
#
# <--Model training -->
from sklearn.naive_bayes import GaussianNB  # được sử dụng để import mô hình Naive Bayes từ thư viện scikit-learn (sklearn)
gnb = GaussianNB()
gnb.fit(X_train, y_train)  # Huấn luyện mô hình
# <--Model training end-->
#
#
#
# <--Dự đoán kết quả-->
y_pred = gnb.predict(X_test)
print(y_pred)
# <--Dự đoán kết quả end-->
#
# <--Kiểm tra điểm chính xác-->
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))  #so sánh class labels y_test(có sẵn) và y_prediction(vừa dùng X_test để dự đoán) với nhau xem giống nhau bao nhiêu %

y_pred_train = gnb.predict(X_train)     #Dự đoán class label y_prediction của training set dựa theo X_train
print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))    #So sánh Y_train (đã có) với Y_pred_train vừa tính)
            #gnb.score(X_train, y_train) tương đương y hệt với accuracy_score(gnb.predict(X_train), y_train)
print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
print(y_test.value_counts())
null_accuracy = (7407 / (7407 + 2362))      #có 7407 <=50k  và   2362 >=50k,tính null accuracy score để xem số điểm chính xác nếu dùng phương pháp luôn dự đoán lớp xuất hiện nhiều nhất, điểm của model nên lớn hơn điểm null accuracy score thì mới nên áp dụng model

print('Null accuracy score: {0:0.4f}'.format(null_accuracy))
# <--Kiểm tra điểm chính xác end-->
#
#
#
#
# <--Ma trận hỗn loạn-->
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0, 0])

print('\nTrue Negatives(TN) = ', cm[1, 1])

print('\nFalse Positives(FP) = ', cm[0, 1])

print('\nFalse Negatives(FN) = ', cm[1, 0])
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])
print(cm_matrix)
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu',linewidths=.5)
plt.show()   # Explicitly display the plot
# <--Ma trận hỗn loạn end-->
#
#
#
#
# <--Chỉ số phân loại-->
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
# <--Chỉ số phân loại end-->
#
#
#
#
# <--Tính xác suất của lớp-->
y_pred_prob = gnb.predict_proba(X_test)[0:10]       # dự đoán phân loại 10 instances đầu tiên của mẫu X_test
print(y_pred_prob)  #kết quả in ra mõi instance sẽ có 2 cột, mỗi cột là khả năng probability của instance đó thuộc về lớp nào

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
print(y_pred_prob_df)

gnb.predict_proba(X_test)[:, 1]         #dự đoán class 1 : Xác suất >50k cho mọi instances
y_pred1 = gnb.predict_proba(X_test)[:, 1]
plt.rcParams['font.size'] = 12
plt.hist(y_pred1, bins=10)
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.xlim(0, 1)
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')
plt.show()
# <--Tính xác suất của lớp end -->
#
# <--ROC - AUC-->
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label='>50K')

plt.figure(figsize=(6, 4))      # xác định chiều dài rộng của biểu đồ

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--')     #(0.0) tới điểm (1,1)

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


#tính điểm ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
#ROC AUC is the percentage of the ROC plot that is underneath the curve. (tính diện tích area ở dưới đường cong ROC,nếu càng cong nhiều thì nó càng gần 1 tức nó tốt,nếu nó sát 0.5 tức gần như đường thẳng thì bộ phân loại này kém
from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))




#10-fold cross-validation should be performed, meaning the training set is divided into 10 subsets, and the model is trained and evaluated 10 times, each time using a different subset as the test set.
#scoring='accuracy': Specifies that the metric used for evaluation is accuracy. The cross_val_score function will return an array of accuracy scores for each fold.
#Hiểu đơn giản là nó train model 5 lần,cách tính điểm scoring = roc_auc rồi nó tính giátrijung bình của 5 lần đó và in ra
from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

from sklearn.model_selection import cross_val_score
#Hiểu đơn giản là nó train model 10 lần và chia data thành 10 subset nhỏ,cách tính điểm scoring = accuracy rồi nó tính giá trijung bình của 10 lần đó và in ra
scores = cross_val_score(gnb, X_train, y_train, cv=10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
