import sqlite3
import csv
import time
from package.faers import dbutils as DBHelper
#新程序
# def main():
#     start_time = time.time()
#
#     db_path = '../faers-data_AGE.sqlite'
#     print("Connected to FAERS database.")
#
#     drugs = parseFile('input/liver-drugs - 1.csv')
#     indications = parseFile('input/immuno-indications.csv')
#
#     print("Getting drug information...")
#     info = DBHelper.getInfo(db_path, drugs, indications)
#
#     if info is None:
#         print("Error: info is None. Cannot generate report.")
#         return
#
#     # print("Generating report...")
#     # DBHelper.generateReport(info, drugs_per_sheet=5)  # 每5个药物一个sheet
#
#     print("Disconnected from FAERS database.")
#
#     end_time = time.time()
#     print(f"Total execution time: {end_time - start_time:.2f} seconds")
#
#
# def parseFile(file=None):
#     if file is None: return False
#     res = dict()
#     with open(file) as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             name = row[0]
#             res[name] = set()
#             for alias in row:
#                 res[name].add(alias.lower())
#     print("Parsed", file)
#     return res
#
#
# if __name__ == "__main__":
#     main()

# 老程序
import sqlite3, csv
import xlsxwriter
from package.faers import dbutils as DBHelper

def main():
    # conn = sqlite3.connect('C:/Users/pengj/OneDrive/桌面/Pregnancy ADR data/FAERS Data/2.5. Pregnancy FAERS ASCII/faers-data.sqlite') 这是老师的路径
    conn = sqlite3.connect('../faers-data_AGE.sqlite')
    c = conn.cursor()
    print("Connected to FAERS database.")
    # -------------------
    # YOUR CODE GOES HERE
    # -------BEGIN-------

    # drugs = parseFile('C:/Users/pengj/OneDrive/桌面/Pregnancy ADR data/FAERS Data/2.5. Pregnancy FAERS ASCII/input/immuno-drugs.csv')彭老师的路径
    # indications = parseFile('C:/Users/pengj/OneDrive/桌面/Pregnancy ADR data/FAERS Data//2.5. Pregnancy FAERS ASCII//input/immuno-indications.csv')
    drugs = parseFile('input/liver-drugs - 1.csv')
    indications = parseFile('input/immuno-indications.csv')
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