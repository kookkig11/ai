import os
from pandas import read_csv
if __name__ == '__main__':
    # path current
    dirpath = os.path.dirname(__file__)
    # dataset information
    filename = "cmc.data"
    # focus wife
    name = ["Age", "Education", "HusbandEducation", "Children", "Religion",
            "Working", "HusbandOccupation", "StandardLiving", "MediaExposure", "Contraceptive"]
    dataset = read_csv(dirpath + "/dataset/" + filename, names = name)
    print(dataset)