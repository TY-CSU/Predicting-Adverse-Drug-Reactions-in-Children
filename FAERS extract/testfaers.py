import sqlite3, csv
import xlsxwriter
from package.faers import dbutils as DBHelper

def main():
    conn = sqlite3.connect('faers-data.sqlite')
    c = conn.cursor()
    print("Connected to FAERS database.")
    # -------------------
    # YOUR CODE GOES HERE
    # -------BEGIN-------

    drugs = parseFile('C:/Users/pengj/OneDrive/桌面/Pregnancy ADR data/FAERS Data/2.5. Pregnancy FAERS ASCII/input/immuno-drugs.csv')
    indications = parseFile('C:/Users/pengj/OneDrive/桌面/Pregnancy ADR data/FAERS Data//2.5. Pregnancy FAERS ASCII//input/immuno-indications.csv')

    info = DBHelper.getInfo(c, drugs, indications)
    DBHelper.generateReport(info)

    # -------END---------
    conn.close()
    print("Disconnected from FAERS database.")

def parseFile(file=None):
    if file is None: return False
    res = dict()
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            name = row[0]
            res[name] = set()
            for alias in row:
                res[name].add(alias.lower())
    print("Parsed", file)
    return res

if __name__ == "__main__":
    main()
