from multiprocessing import Pool
import numpy as np
import time
import pandas as pd
import os
from numpy import array
import sys
sys.path.append('../serialize/')
from log_to_sequence import log_to_seq
from trace_to_sequence import trace_to_seq
sys.path.append('../')
import config
import json
import re
from collections import OrderedDict

data_path = '../data/'
store_linear_interpolation_data = '../update_linear_interpolation_data/'
store_serilize_data = '../serilize_data'
ground_truth_path = '../labeled_service/'

if not os.path.exists(store_serilize_data):
    os.mkdir(store_serilize_data)
if not os.path.exists(store_serilize_data + '/metrics'):
    os.mkdir(store_serilize_data + '/metrics')
if not os.path.exists(store_serilize_data + '/log'):
    os.mkdir(store_serilize_data + '/log')
if not os.path.exists(store_serilize_data + '/trace'):
    os.mkdir(store_serilize_data + '/trace')


# def alignment(service_s, metric, log, trace):
#     metric = metric[(metric['timestamp']>=config.start_time[service_s]) & (metric['timestamp']<=config.end_time[service_s])]
#     log = log[(log['timestamp']>=config.start_time[service_s]) & (log['timestamp']<=config.end_time[service_s])]
#     trace = trace[(trace['timestamp']>=config.start_time[service_s]) & (trace['timestamp']<=config.end_time[service_s])]
    
#     metric = metric.drop_duplicates(['timestamp'])
#     trace = trace.drop_duplicates(['timestamp'])
#     log = log.drop_duplicates(['timestamp'])
    
#     trace = trace[trace['timestamp'].isin(metric.timestamp.values)]
#     log = log[log['timestamp'].isin(metric.timestamp.values)]
    
#     while trace.shape[0]!=log.shape[0] or log.shape[0]!=metric.shape[0] or trace.shape[0]!=metric.shape[0]:
#         min_length = min([metric.shape[0], trace.shape[0], log.shape[0]])
#         if trace.shape[0] == min_length:
#             print('1--')
#             log = log[log['timestamp'].isin(trace.timestamp.values)]
#             metric = metric[metric['timestamp'].isin(trace.timestamp.values)]
#         elif log.shape[0] == min_length:
#             print('2--')
#             metric = metric[metric['timestamp'].isin(log.timestamp.values)]
#             trace = trace[trace['timestamp'].isin(log.timestamp.values)]
#         else:
#             print('3--')
#             log = log[log['timestamp'].isin(metric.timestamp.values)]
#             trace = trace[trace['timestamp'].isin(metric.timestamp.values)]
        
#     metric = metric.reset_index(drop=True)
#     log = log.reset_index(drop=True)
#     trace = trace.reset_index(drop=True)
    
#     return log, metric, trace
    
def alignment(service_s, metric, log, trace):
    # Фильтрация по времени
    metric = metric[(metric['timestamp'] >= config.start_time[service_s]) & (metric['timestamp'] <= config.end_time[service_s])]
    log = log[(log['timestamp'] >= config.start_time[service_s]) & (log['timestamp'] <= config.end_time[service_s])]
    trace = trace[(trace['timestamp'] >= config.start_time[service_s]) & (trace['timestamp'] <= config.end_time[service_s])]
    
    # Убираем дубликаты по timestamp
    metric = metric.drop_duplicates(['timestamp'])
    log = log.drop_duplicates(['timestamp'])
    trace = trace.drop_duplicates(['timestamp'])

    # Находим общее множество timestamp
    all_timestamps = sorted(set(metric['timestamp']) | set(log['timestamp']) | set(trace['timestamp']))
    timestamps_df = pd.DataFrame({'timestamp': all_timestamps})

    # Объединяем данные по timestamp (outer join)
    metric_aligned = timestamps_df.merge(metric, on='timestamp', how='left')
    log_aligned = timestamps_df.merge(log, on='timestamp', how='left')
    trace_aligned = timestamps_df.merge(trace, on='timestamp', how='left')

    # Заполняем пропущенные значения нулями (или другими значениями при необходимости)
    metric_aligned = metric_aligned.fillna(0)
    log_aligned = log_aligned.fillna(0)
    trace_aligned = trace_aligned.fillna(0)

    return log_aligned.reset_index(drop=True), metric_aligned.reset_index(drop=True), trace_aligned.reset_index(drop=True)

def write_log_csv(service_s, name, temp, stru):
    stru = stru.drop_duplicates(['timestamp'])
    stru['timestamp'] = stru['timestamp'].astype(int)
    print("stru:", stru)
    temp = temp[temp['EventTemplate'].notnull()]
    temp = temp[temp['EventTemplate'].str.contains('INFO')]
    print("temp:", temp)
    log_series = dict({i:[] for i in temp['EventTemplate'].values.tolist()})
    log_series['total_log_length'] = []
    num_cores = len(log_series.keys())
    pool = Pool(processes=num_cores)  # multi-threading
    data_subsets = []
    for i in range(num_cores):
        key_name = list(log_series.keys())[i]
        data_subsets.append(pool.apply_async(log_to_seq, args=(key_name, temp, stru, config.start_time[service_s], config.end_time[service_s])))

    pool.close()
    pool.join()
    
    results = pd.DataFrame()
    count = 0
    for res in data_subsets:
        data = res.get()
        data_timestamp = []
        data_num = []
        for d in data:
            data_timestamp.append(d[0])
            data_num.append(d[1])
        results.loc[:, str(count)] = pd.Series(data_num)
        results.loc[:, "timestamp"] = pd.Series(data_timestamp)
        count += 1
    output_dir = os.path.join(store_serilize_data, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(store_serilize_data + '/logs/' + name + '.csv', index=False)


# def write_trace_json(service_s, name, data):
#     num_cores = 100
#     pool = Pool(processes=num_cores)
#     trace_data = []
#     data_subsets = []
#     begin = 0
#     for j in range(num_cores):
#         data_subsets.append(data[begin:begin+int(data.shape[0]//num_cores)])
#         begin = begin+int(data.shape[0]//num_cores)
#     print("data subsets created")
#     for i in range(num_cores):
#         trace_data.append(pool.apply_async(trace_to_seq, args=(data_subsets[i], config.start_time[service_s], config.end_time[service_s])))
#     pool.close()
#     pool.join()
#     results = OrderedDict()
#     for res in trace_data:
#         results.update(res.get())
#     trace_data = results.copy()
#     trace_dict = {
#         'version': "1.0",
#         'results': trace_data,
#         'explain': {
#             'used': True,
#             'details': "this is for josn test",
#         }
#     }

#     json_str = json.dumps(trace_dict, indent=4)
#     with open(store_serilize_data + '/traces/' + name, 'w') as json_file:
#         json_file.write(json_str)
        
#     print('*'*100)

import os
import json
from collections import OrderedDict

def write_trace_json(service_s, name, data):
    print("Starting trace conversion in single-threaded mode...")

    try:
        # Вызов функции trace_to_seq на всём наборе данных
        trace_data = trace_to_seq(
            data,
            config.start_time[service_s],
            config.end_time[service_s]
        )

        # Убедимся, что результат — словарь
        if not isinstance(trace_data, dict):
            print("[ERROR] trace_to_seq вернул не словарь.")
            return

        trace_dict = {
            'version': "1.0",
            'results': trace_data,
            'explain': {
                'used': True,
                'details': "this is for json test",
            }
        }

        # Создание директории, если её нет
        output_dir = os.path.join(store_serilize_data, 'traces')
        os.makedirs(output_dir, exist_ok=True)

        # Сохранение в файл
        json_path = os.path.join(output_dir, name)
        with open(json_path, 'w') as json_file:
            json.dump(trace_dict, json_file, indent=4)

        print(f"✔ JSON сохранён: {json_path}")

    except Exception as e:
        print(f"[ERROR] Во время сохранения произошла ошибка: {e}")



def generate_sequence_data(name):
    # log data
    log_dataset = pd.read_csv(store_serilize_data + '/logs/' + name + '.csv')
    log_dataset = linear_interpolation(name, log_dataset, 1)
    output_dir = os.path.join(store_linear_interpolation_data, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    log_dataset.to_csv(store_linear_interpolation_data + '/logs/' + name + '.csv', index=False)
    print("log_dataset.shape: ", log_dataset.shape)

    # trace data
    trace_json = pd.read_json(store_serilize_data + '/traces/' + name + '.json', typ='series')
    results = trace_json['results'] if 'results' in trace_json else {}
    trace_rows = []
    for ts_str, values in results.items():
        if not isinstance(values, (list, tuple)):
            continue
        row = {str(idx): val for idx, val in enumerate(values)}
        try:
            row['timestamp'] = int(ts_str)
        except ValueError:
            continue
        trace_rows.append(row)
    trace_dataset = pd.DataFrame(trace_rows)
    trace_dataset = linear_interpolation(name, trace_dataset, 1)
    trace_dataset.to_csv(store_linear_interpolation_data + '/trace/' + name + '.csv', index=False)
    print("trace_dataset.shape: ", trace_dataset.shape)


def fill_missing_range(df, field, range_from, range_to, range_step=1, fill_with=0):
    return df.merge(how='right', on=field,
            right = pd.DataFrame({field:np.arange(range_from, range_to, range_step)}))\
                .sort_values(by=field).reset_index().fillna(fill_with).drop(['index'], axis=1)
      

def linear_interpolation(service_s, data, interval):
    data['timestamp'] = data['timestamp'].astype('int')
    data = data.drop_duplicates(['timestamp'])
    print("Raw data:", data)
    data = fill_missing_range(data, 'timestamp', config.start_time[service_s], config.end_time[service_s], interval, np.nan)
    data = data.interpolate()
    print("After data:", pd.DataFrame(data))
    return data
    

def write(service_s, store_path):
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    df = pd.read_csv(data_path + service_s + '_2021-07-01_2021-07-15.csv')
    print(df.head(5))
    # Метрики приходят в мс, приводим к секундам, чтобы совпасть с логами/трейсами
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    if df['timestamp'].dropna().max() > 1e11:
        df['timestamp'] = (df['timestamp'] / 1000).astype('int')
    else:
        df['timestamp'] = df['timestamp'].astype('int')
    df = df.drop_duplicates(['timestamp'])
    print(df)
    df = df[(df['timestamp']>=config.start_time[service_s]) & (df['timestamp']<=config.end_time[service_s])]
    if 'ground_truth' in df.columns:
        df = df.drop(['ground_truth'], axis=1)
    if 'label' in df.columns:
        df = df.drop(['label'], axis=1)
    if 'run_fault' in df.columns:
        df = df.drop(['run_fault'], axis=1)
    df = df.loc[:, (df != 0).any(axis=0)]
    df.to_csv(store_linear_interpolation_data + '/metrics/' + service_s + '.csv', index=False)
    
    print("df ok (metrics)")
    trace = pd.read_csv(data_path + 'trace_table_' + service_s.split('_')[0] + '_2021-07.csv')
    print("traces read")
    # timestamp в трейсе в формате строки даты — переводим в секундный unix-тайм
    trace['timestamp'] = pd.to_datetime(trace['timestamp'], errors='coerce')
    trace = trace.dropna(subset=['timestamp'])
    trace['timestamp'] = (trace['timestamp'].astype('int64') // 10**9).astype(int)
    trace = trace[(trace['timestamp']>=config.start_time[service_s]) & (trace['timestamp']<=config.end_time[service_s])]
    trace = trace.drop_duplicates(['timestamp'])
    print("traces dropped dublicates")
    if not os.path.exists(store_path + '/trace/'):
        os.mkdir(store_path + '/trace/')
        print("dir ctreated")
    print("stert writing trace to json")
    write_trace_json(service_s, service_s.split('_')[0] + '.json', trace)
    print("trace converted to json")
   
    stru = pd.read_csv(data_path + service_s.split('_')[0] + '_stru.csv')
    temp = pd.read_csv(data_path + service_s.split('_')[0] + '_temp.csv')
    print("stru.timestamp:", stru['timestamp'].values[0], stru['timestamp'].values[-1])
    write_log_csv(service_s, service_s.split('_')[0], temp, stru)
    print("logs to csv")

     
def get_channels(service_s, dirname, proportion):
    write(service_s, store_linear_interpolation_data)
    generate_sequence_data(service_s.split('_')[0])
    
    trace = pd.read_csv(store_linear_interpolation_data + '/trace/' + service_s +'.csv')
    stru = pd.read_csv(store_linear_interpolation_data + '/logs/' + service_s +'.csv')
    metric = pd.read_csv(store_linear_interpolation_data + '/metrics/' + service_s +'.csv')
    label_with_time = pd.read_csv(config.label_path + service_s + '.csv')
 
    align_log, align_metric, align_trace = alignment(service_s, metric, stru, trace)
    label_with_time = label_with_time[label_with_time['timestamp'].isin(align_metric['timestamp'].values)]
    print("align_metric:", align_metric.shape)
    print("align_log:", align_log.shape)
    print("align_trace:", align_trace.shape)
    if 'train' in dirname:
        print('get_channels() train')
        align_metric = align_metric[:int(proportion*align_metric.shape[0])]
        align_log = align_log[:int(proportion*align_log.shape[0])]
        align_trace = align_trace[:int(proportion*align_trace.shape[0])]
        label_with_time = label_with_time[:int(proportion*len(label_with_time))]
        # timestamp = timestamp[:int(proportion*len(timestamp))]
    elif 'test' in dirname:
        print('get_channels() test')
        align_metric = align_metric[int(proportion*align_metric.shape[0]):]
        align_log = align_log[int(proportion*align_log.shape[0]):]
        align_trace = align_trace[int(proportion*align_trace.shape[0]):]
        label_with_time = label_with_time[int(proportion*len(label_with_time)):]
        # timestamp = timestamp[int(proportion*len(timestamp)):]
        
    align_metric = align_metric.drop('timestamp', axis=1)
    align_log = align_log.drop('timestamp', axis=1)
    align_trace = align_trace.drop('timestamp', axis=1)
    print(align_metric, align_log, align_trace)
    return label_with_time, align_metric, align_log, align_trace
