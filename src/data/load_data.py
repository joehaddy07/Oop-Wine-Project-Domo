# ============================================================
# WINE DATA LOADER
# File:
#     src/data/load_data.py
#
# PURPOSE:
#     This class is responsible for loading our wine dataset.
# ============================================================


# ============================================================
# IMPORT PANDAS
# ============================================================
# pandas is used to read and work with CSV data.
# ============================================================

import pandas as pd


# ============================================================
# CLASS: WineDataLoader
# ============================================================
# A CLASS is a blueprint.
#
# Think of the class as a blueprint for creating objects
# that know how to load wine data.
# ============================================================

class WineDataLoader:

    # ========================================================
    # __init__()
    # ========================================================
    # __init__() is a special Python method.
    #
    # It runs automatically when we create an object
    # from the WineDataLoader class.
    #
    # self:
    #     Represents the object being created.
    #
    # file_path:
    #     The location of our wine.csv file.
    # ========================================================

    def __init__(self, file_path):

        # ====================================================
        # ATTRIBUTE: self.file_path
        # ====================================================
        # We store the file path inside the object.
        #
        # self.file_path means:
        #
        # "The file_path belonging to THIS object."
        # ====================================================

        self.file_path = file_path


    # ========================================================
    # METHOD: load_data()
    # ========================================================
    # A METHOD is a function that belongs to a class.
    #
    # This method will read our CSV file.
    # ========================================================

    def load_data(self):

        # ====================================================
        # Read the CSV file.
        #
        # self.file_path retrieves the path that we stored
        # when the object was created.
        # ====================================================

        wine_dataset = pd.read_csv(self.file_path)

        # ====================================================
        # Return the DataFrame.
        # ====================================================

        return wine_dataset