import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree # create model
from sklearn import metrics

if __name__ == '__main__':
    filename = "cmc.data"
    # path current
    dirpath = os.path.dirname(__file__)
    # focus wife
    name = ["Age", "Education", "HusbandEducation", "Children", "Religion",
            "Working", "HusbandOccupation", "StandardLiving", "MediaExposure", "Contraceptive"]
    dataset = pd.read_csv(dirpath + "/dataset/" + filename, names = name)
    dataArr = dataset.values
    
    x = dataArr[:,0:9]
    y = dataArr[:,9]    # target
    
    # 60% training and 40% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Accuracy
    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))



    
#     print(confusion_matrix(y_test, y_pred))
#     print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#     dot_data = export_graphviz(clf,out_file='tree_limited.dot',filled=True,rounded=True,special_characters=True)
#     print(dot_data)
