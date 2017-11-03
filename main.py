#!/usr/bin/python3.5
import numpy as np
import pandas as pa
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    work = pa.read_csv('Work.csv')
    data = []
    y_true = []
    for i in range(1, len(work)):
        x = work.iloc[i]
        y = work.iloc[i - 1]
        if x['ID'] == y['ID']:
            user = []
            # user.append(x['ID'])
            user.append(x['Psicho'])
            user.append(x['Bagrot'])
            if x['Course'] == 1:
                user.append(x['Units'])
                user.append(x['Grade'])
                user.append(y['Units'])
                user.append(y['Grade'])
            else:
                user.append(y['Units'])
                user.append(y['Grade'])
                user.append(x['Units'])
                user.append(x['Grade'])
            user.append(x['Gender'])
            user.append(x['Age'])
            user.append(x['Amount'])
            # user.append(x['Group'])
            if x['y'] > 85:
                y_true.append(1)
            else:
                y_true.append(0)
            data.append(user)

    # pred_train, pred_test, tar_train, tar_test = train_test_split(data, y_true, test_size=.4)

    # print(pred_train)
    count_lists = np.zeros(9)
    weight_list = np.zeros(9)
    for i in range(50):
        print(i)
        regr = RandomForestClassifier(n_estimators=200)
        regr.fit(data, y_true)
        # print(regr.score(pred_test, tar_test))
        # predict_y = regr.predict(data)
        # print(tar_test)
        # print(predict_y)
        # print(confusion_matrix(y_true, predict_y))
        # print(regr.feature_importances_)
        my_list = regr.feature_importances_
        weight_list += my_list
        temp = [x for x in my_list]
        sorted_elements = np.zeros(9)
        for i in range(len(temp)):
            idx = np.argmax(temp)
            sorted_elements[idx] = len(temp) - i
            temp[idx] = 0

        for j, element in enumerate(sorted_elements):
            count_lists[j] += element

    count_lists /= float(50)
    weight_list /= float(50)

    print(count_lists)
    print(weight_list)