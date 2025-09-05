# from multiprocessing import Pool
# import pandas as pd
# import time
# import math
# from collections import Counter
# from timeit import default_timer as timer
# from package.utils import progressbar as prog
# from package.faers import queryhelper as sqlh
# from package.faers import signal_scores as ss
# from contextlib import closing
# import sqlite3
# from multiprocessing import Pool
# from contextlib import closing
# from timeit import default_timer as timer
# import multiprocessing
# import itertools
# import pandas as pd
# import time
# from timeit import default_timer as timer
# import sqlite3
# from multiprocessing import Pool
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed


# def getInfo(db_path, drugmap, indicationmap, drugs_per_sheet=10):
#     start = timer()
#     info = {}  # 初始化 info 字典
#     print("Scanning adverse events...")
#
#     with sqlite3.connect(db_path) as conn:
#         c = conn.cursor()
#         aeReference = scanAdverseEvents(c)
#     aeMap, aeCounter = aeReference
#
#     print("Processing drugs in parallel and generating report...")
#
#     filename = f"./output/results_{time.strftime('%Y-%m-%d_%H%M%S')}.xlsx"
#     print(f"Saving report to {filename}")
#
#     with pd.ExcelWriter(filename, engine='openpyxl') as writer:
#         df_drug = pd.DataFrame(columns=["Drug", "Indication", "Entries"])
#
#         with ThreadPoolExecutor() as executor:
#             futures = []
#             for drug, names in drugmap.items():
#                 futures.append(executor.submit(process_drug, db_path, drug, names, indicationmap, aeMap, aeCounter))
#
#             for i, future in enumerate(as_completed(futures)):
#                 sheet_index = i // drugs_per_sheet
#                 if i % drugs_per_sheet == 0:
#                     df_drugInfo = pd.DataFrame(
#                         columns=["Drug", "Indication", "Adverse Event", "Reports", "Frequency", "PRR", "ROR",
#                                  "CI (Lower 95%)",
#                                  "CI (Upper 95%)", "CI < 1"])
#
#                 drug_info = future.result()
#                 info.update(drug_info)  # 将处理后的药物信息添加到 info 字典中
#
#                 for drug, indications in drug_info.items():
#                     df_drugInfo, df_drug = process_drug_results(drug, indications, df_drugInfo, df_drug)
#
#                 if (i + 1) % drugs_per_sheet == 0 or i == len(futures) - 1:
#                     if not df_drugInfo.empty:
#                         sheet_name = f"Drugs_{sheet_index * drugs_per_sheet + 1}-{min((sheet_index + 1) * drugs_per_sheet, len(drugmap))}"
#                         df_drugInfo.to_excel(writer, sheet_name, index=False)
#
#         if not df_drug.empty:
#             df_drug.to_excel(writer, "Drug Count", index=False)
#
#     end = timer()
#     print(f"All drugs processed and report generated in {end - start} seconds.")
#
#     return info  # 返回包含所有处理后药物信息的字典
#
# def process_drug_results(drug, indications, df_drugInfo, df_drug):
#     for indi, data in indications.items():
#         num_reports = len(data['pids'])
#         df_drug = pd.concat([df_drug, pd.DataFrame({"Drug": [drug], "Indication": [indi], "Entries": [num_reports]})],
#                             ignore_index=True)
#
#         for ae, count in data['aes'].items():
#             freq = count / num_reports if num_reports else 0
#             prr = data['stats'][ae]['PRR']
#             ror = data['stats'][ae]['ROR']
#             ci_valid = (ror[2] - ror[1] < 1) if isinstance(ror, (list, tuple)) and len(ror) > 2 else False
#
#             new_row = pd.DataFrame({
#                 "Drug": [drug], "Indication": [indi], "Adverse Event": [ae],
#                 "Reports": [count], "Frequency": [freq], "PRR": [prr],
#                 "ROR": [ror[0] if isinstance(ror, (list, tuple)) else ror],
#                 "CI (Lower 95%)": [ror[1] if isinstance(ror, (list, tuple)) and len(ror) > 1 else None],
#                 "CI (Upper 95%)": [ror[2] if isinstance(ror, (list, tuple)) and len(ror) > 2 else None],
#                 "CI < 1": [ci_valid]
#             })
#             df_drugInfo = pd.concat([df_drugInfo, new_row], ignore_index=True)
#
#     return df_drugInfo, df_drug
#
#
# def process_drug(db_path, drug, names, indicationmap, aeMap, aeCounter):
#     try:
#         info = {drug: {}}
#         print(f"Processing drug: {drug}")
#
#         with sqlite3.connect(db_path) as conn:
#             c = conn.cursor()
#             info[drug]['all'] = getDrugInfo(c, aeMap, names)
#             info[drug]['all']['stats'] = getAEStats(aeCounter, info[drug]['all']['aes'])
#
#             for indi, indi_pts in indicationmap.items():
#                 info[drug][indi] = getDrugInfoByIndication(c, aeMap, names, indi_pts)
#                 info[drug][indi]['stats'] = getAEStats(aeCounter, info[drug][indi]['aes'])
#         return info
#     except Exception as e:
#         print(f"Error processing drug {drug}: {str(e)}")
#         return {drug: {}}
# def getDrugInfo(c, aeMap, drugnames):
#     PIDs, AEs = set(), Counter()
#     query = sqlh.selectDrug(drugnames)
#     c.execute(query)
#     for primaryid, in c.fetchall():
#         PIDs.add(primaryid)
#         pid = str(primaryid)
#         if pid in aeMap:
#             AEs.update(aeMap[pid])
#     return {'pids': PIDs, 'aes': AEs}
#
#
# def getDrugInfoByIndication(c, aeMap, drugnames, indications):
#     PIDs, AEs = set(), Counter()
#     drugNameQuery = sqlh.selectDrug(drugnames)
#     indicationQuery = sqlh.selectIndication(indications)
#     query = f"{drugNameQuery} INTERSECT {indicationQuery}" if indicationQuery else drugNameQuery
#     c.execute(query)
#     for primaryid, in c.fetchall():
#         PIDs.add(primaryid)
#         pid = str(primaryid)
#         if pid in aeMap:
#             AEs.update(aeMap[pid])
#     return {'pids': PIDs, 'aes': AEs}
#
#
# def scanAdverseEvents(c):
#     aeMap, aeCounter = {}, Counter()
#     c.execute("SELECT COUNT(*) FROM REACTION")
#     total = c.fetchone()[0]
#     c.execute("SELECT IFNULL(primaryid, isr), pt FROM REACTION")
#     for i, (primaryid, pt) in enumerate(c):
#         primaryid = str(primaryid).lower()
#         pt = str(pt).lower().replace('\n', '')
#         aeCounter[pt] += 1
#         aeMap.setdefault(primaryid, set()).add(pt)
#         if i % 20000 == 0:
#             prog.update("Scanning adverse events", i / total)
#     prog.update("Scanning adverse events", 1)
#     return aeMap, aeCounter
#
#
# def getAEStats(totalAEs, drugAEs):
#     sum_totalAE = sum(totalAEs.values())
#     sum_drugAE = sum(drugAEs.values())
#     stats = {}
#     for ae in drugAEs:
#         var_A = drugAEs[ae]
#         var_B = sum_drugAE - var_A
#         var_C = totalAEs[ae] - var_A
#         var_D = sum_totalAE - var_A - var_B - var_C
#         stats[ae] = {
#             'PRR': ss.getPRR(var_A, var_B, var_C, var_D),
#             'ROR': ss.getROR(var_A, var_B, var_C, var_D)
#         }
#     return stats
#
#
# def generateReport(info, drugs_per_sheet=2):
#     if info is None:
#         print("Error: info is None. Cannot generate report.")
#         return
#
#     start = timer()
#     print("Generating report")
#
#     filename = f"./output/results_{time.strftime('%Y-%m-%d_%H%M%S')}.xlsx"
#     print(f"Saving report to {filename}")
#
#     try:
#         with pd.ExcelWriter(filename, engine='openpyxl') as writer:
#             df_drug = pd.DataFrame(columns=["Drug", "Indication", "Entries"])
#
#             if not info:  # 如果 info 为空字典
#                 df_empty = pd.DataFrame(columns=["No Data"])
#                 df_empty.to_excel(writer, "Empty Sheet", index=False)
#             else:
#                 for sheet_index, drug_chunk in enumerate(chunked(info.items(), drugs_per_sheet)):
#                     df_drugInfo = pd.DataFrame(
#                         columns=["Drug", "Indication", "Adverse Event", "Reports", "Frequency", "PRR", "ROR",
#                                  "CI (Lower 95%)",
#                                  "CI (Upper 95%)", "CI < 1"])
#
#                     for drug, indications in drug_chunk:
#                         for indi, data in indications.items():
#                             # 这里应该是处理数据的代码
#                             # 确保正确填充 df_drugInfo 和 df_drug
#                             pass  # 如果没有具体代码，可以使用 pass
#
#                     sheet_name = f"Drugs_{sheet_index * drugs_per_sheet + 1}-{(sheet_index + 1) * drugs_per_sheet}"
#                     if not df_drugInfo.empty:
#                         df_drugInfo.to_excel(writer, sheet_name, index=False)
#
#             if not df_drug.empty:
#                 df_drug.to_excel(writer, "Drug Count", index=False)
#
#             # 确保至少有一个工作表
#             if not writer.sheets:
#                 df_empty = pd.DataFrame(columns=["No Data"])
#                 df_empty.to_excel(writer, "Empty Sheet", index=False)
#
#         end = timer()
#         print(f"Report generated in {end - start} seconds.")
#
#     except Exception as e:
#         print(f"An error occurred while generating the report: {str(e)}")
#
#     return filename  # 返回生成的文件名
#
# def chunked(iterable, n):
#     """Yield successive n-sized chunks from iterable."""
#     it = iter(iterable)
#     while True:
#         chunk = tuple(itertools.islice(it, n))
#         if not chunk:
#             return
#         yield chunk

from collections import Counter
import numpy as np
import pandas as pd
import sys
import cmath
import math
import time
import xlsxwriter
from package.utils import progressbar as prog
from package.faers import queryhelper as sqlh
from package.faers import signal_scores as ss
from timeit import default_timer as timer
# info
# --[drug] drug (dict)
#   --['all'] all indications (dict)
#     --['pids'] primaryids (list)
#     --['aes'] adverse events (counter)
#     --['stats'] stats (dict)
#       --[ae] each AE (dict)
#         --['PRR']
#         --['ROR']
#   --[indi] each indication (dict)
#     --['pids'] primaryids (list)
#     --['aes'] adverse events (counter)
#     --['stats'] stats (dict)
#       --[ae] each AE (dict)
#         --['PRR']
#         --['ROR']
def getInfo(c, drugmap, indicationmap):
    start = timer()

    aeReference = scanAdverseEvents(c)
    aeMap = aeReference[0]
    aeCounter = aeReference[1]

    num_drugs = len(drugmap)
    num_indis = len(indicationmap)
    print("Searching database")
    drugcounter = 0
    info = dict()
    for drug, names in drugmap.items():
        drugcounter += 1
        print("--Drug (" + str(drugcounter) + "/" + str(num_drugs) + "):", drug)
        info[drug] = dict()
        print("  --All Indications")
        info[drug]['all'] = getDrugInfo(c, aeMap, names)
        print("    --primaryids:", len(info[drug]['all']['pids']))
        print("    --adverse events: done")
        info[drug]['all']['stats'] = getAEStats(aeCounter, info[drug]['all']['aes'])
        print("    --stats: done")
        indicounter = 0
        for indi, indi_pts in indicationmap.items():
            indicounter += 1
            print("  --Indication (" + str(indicounter) + "/" + str(num_indis) + "):", indi)
            info[drug][indi] = getDrugInfoByIndication(c, aeMap, names, indi_pts)
            print("    --primaryids:", len(info[drug][indi]['pids']))
            print("    --adverse events: done")
            info[drug][indi]['stats'] = getAEStats(aeCounter, info[drug][indi]['aes'])
            print("    --stats: done")
    end = timer()
    print("Completed in", (end - start), "seconds.")
    return info


def getDrugInfo(c, aeMap, drugnames):
    PIDs, AEs = [], Counter()
    query = sqlh.selectDrug(drugnames)
    c.execute(query)
    for i in c:
        primaryid = i[0]
        PIDs.append(primaryid)
        pid = str(primaryid)
        if pid in aeMap:
            for ae in aeMap[pid]:
                AEs[ae] += 1
    info = dict()
    info['pids'] = PIDs
    info['aes'] = AEs

    return info


def getAEStats(totalAEs, drugAEs):
    sum_totalAE = sum(totalAEs.values())
    stats = dict()
    for ae in drugAEs:
        stats[ae] = dict()
        sum_drugAE = sum(drugAEs.values())
        var_A = drugAEs[ae]  # Event Y for Drug X
        var_B = sum_drugAE - var_A  # Other events for Drug X
        var_C = totalAEs[ae] - var_A  # Event Y for other drugs
        var_D = sum_totalAE - var_A - var_B - var_C  # Other events for other drugs\
        stats[ae]['PRR'] = ss.getPRR(var_A, var_B, var_C, var_D)
        stats[ae]['ROR'] = ss.getROR(var_A, var_B, var_C, var_D)
    return stats


# Given specified drugnames / indications
# Return
#   --[pids]: List of primaryIDs for the combo of drugname / indication
#   --[aes]: Counter of
def getDrugInfoByIndication(c, aeMap, drugnames, indications):
    PIDs, AEs = [], Counter()
    drugNameQuery = sqlh.selectDrug(drugnames)
    indicationQuery = sqlh.selectIndication(indications)
    query = drugNameQuery
    if not indicationQuery is False: query = query + " INTERSECT " + indicationQuery
    c.execute(query)
    for i in c:
        primaryid = i[0]
        PIDs.append(primaryid)
        pid = str(primaryid)
        if pid in aeMap:
            for ae in aeMap[pid]: AEs[ae] += 1
    info = dict()
    info['pids'], info['aes'] = PIDs, AEs
    return info


# Returns the following objects
# aeMap: set of preferred terms specified in each primaryid
# aeCounter: counter with frequencies of all preferred terms
def scanAdverseEvents(c):
    prog.update("Scanning adverse events", 0)
    start, aeMap, aeCounter = timer(), dict(), Counter()
    c.execute("SELECT COUNT(*) FROM REACTION")
    counter, total = 0, c.fetchone()[0]
    c.execute("SELECT IFNULL(primaryid, isr), pt FROM REACTION")
    for i in c:
        primaryid = str(i[0]).lower()
        pt = str(i[1]).lower().replace('\n', '')
        aeCounter[pt] += 1
        if primaryid in aeMap:
            aeMap[primaryid].add(pt)
        else:
            aeMap[primaryid] = set([pt]); counter += 1
        if counter % 20000 == 0: prog.update("Scanning adverse events", (counter / total))
    end = timer()
    prog.update("Scanning adverse events", 1)
    print("Completed in", (end - start), "seconds.")
    return (aeMap, aeCounter)


def getFreq(reports, total):
    if reports == 0 or total == 0:
        return 0
    else:
        return float(reports) / float(total)


# Returns timestamp filename
def getOutputFilename(extension):
    timestr = time.strftime("results_%Y-%m-%d_%H%M%S")
    return (timestr + extension)


# count the adverse events in a specific iterable of primaryIDs
def countAdverseEvents(aeMap, primaryids):
    aeCounts = Counter()
    primaryids = set(primaryids)
    for primaryid in primaryids:
        pid = str(primaryid)
        if pid in aeMap:
            for ae in aeMap[pid]:
                aeCounts[ae] += 1
    return aeCounts


def generateReport(info):
    start = timer()
    print("Generating report")
    df_drugInfo = pd.DataFrame(
        columns=["Drug", "Indication", "Adverse Event", "Reports", "Frequency", "PRR", "ROR", "CI (Lower 95%)",
                 "CI (Upper 95%)", "CI < 1"])
    df_drug = pd.DataFrame(columns=["Drug", "Indication", "Entries"])
    drugcounter = 0
    num_drugs = len(info)
    for drug, indications in info.items():
        drugcounter += 1
        msg = "--Drug (" + str(drugcounter) + "/" + str(num_drugs) + "): " + drug
        print(msg)
        total_reports = len(info[drug]['all']['pids'])
        indicounter = 0
        num_indis = len(indications)
        for indi, data in indications.items():
            indicounter += 1
            msg = "  --Indication (" + str(indicounter) + "/" + str(num_indis) + "): " + indi
            num_reports = len(info[drug][indi]['pids'])
            df_drug.loc[len(df_drug)] = [drug, indi, num_reports]
            AEs = data['aes']
            aecounter = 0
            total_AEs = sum(AEs.values())
            for ae in AEs:
                aecounter += AEs[ae]
                freq = getFreq(AEs[ae], num_reports)
                prr = data['stats'][ae]['PRR']
                ror = data['stats'][ae]['ROR']
                ci_valid = False
                try:
                    if ((ror[2] - ror[1]) < float(1)):
                        ci_valid = True
                except:
                    ci_valid = False
                df_drugInfo.loc[len(df_drugInfo)] = [drug, indi, ae, AEs[ae], freq, prr, ror[0], ror[1], ror[2],
                                                     ci_valid]
                prog.update(msg, aecounter / float(total_AEs))
    filename = getOutputFilename(".xlsx")
    filename = "./output/" + filename
    print("Saving report to", filename)
    writer = pd.ExcelWriter(filename)
    df_drugInfo.to_excel(writer, "Drug Info")
    df_drug.to_excel(writer, "Drug Count")
    writer.close()
    end = timer()
    print("Completed in", (end - start), "seconds.")