import os
import pydotplus
from subprocess import check_call
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz # create model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from IPython.core.display import Image

if __name__ == '__main__':
    filename = "cmc.data"
    # path current
    dirpath = os.path.dirname(__file__)
    # focus wife
    name = ["Age", "Education", "HusbandEducation", "Children", "Religion",
            "Working", "HusbandOccupation", "StandardLiving", "MediaExposure", "Contraceptive"]
    dataset = read_csv(dirpath + "/dataset/" + filename, names = name)
    dataArr = dataset.values
    x = dataArr[:,0:9]
    y = dataArr[:,9]    # target
    
    # 70% training and 30% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))

    # Accuracy
    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))

    dot_data = StringIO()
    export_graphviz(clf, out_file = 'contra_tree.dot', feature_names=name[0:9],
                        filled = True, rounded = True, special_characters = True)
    # export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=name[0:9], class_names=[1,2,3])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('contra_tree.png')
    # Image(graph.create_png())
    export_graphviz(clf, out_file = 'contra_tree.dot', feature_names=name[0:9],
                        filled = True, rounded = True, special_characters = True)
    check_call(['dot', '-Tpng', dirpath + "/contra_tree.dot", '-o', 'contra_tree.png'])
    Image(filename = dirpath + '/contra_tree.png')