label_path = '../labeled_service/'

# Set time window to actual data ranges present in metric/log/trace CSVs
# mobservice2_2021-07-01_2021-07-15.csv:      1625104811 .. 1625105891
# trace_table_mobservice2_2021-07.csv:        1625079263 .. 1625079281
# mobservice2_stru.csv (logs, first 7 days):  1625136862 .. 1625702399
# metrics converted to seconds start around:   1625133601
# One-day window (24h) from the metric start:  1625133601 .. 1625220000
start_time = {'mobservice2': 1625133601}
end_time = {'mobservice2': 1626278397}
