import xlwt
import itertools
import numpy as np
from os import path, mkdir, listdir

save_path = 'resuts_xls'
if not path.exists(save_path):
    mkdir(save_path)


# Need to pass the args dictionary
#   - run.py passes it or - can load it from args directory
#   - if none, loads it from a specified folder
def write_results(args, folder=None):

    book = xlwt.Workbook(encoding='utf-8')

    sheets = {}
    metric_names = ['bae', 'accuracy', 'micro_precision', 'micro_recall', 'micro_f1',
            'macro_precision', 'macro_recall', 'macro_f1', 'hamming_loss', 'coverage', 'pak', 'ranking_loss',
            'average_precision', 'val_loss']
    n_metrics = len(metric_names)
    cols = ['', 'labels', 'percents', 'folds', ''] + metric_names

    if args is not None:
        cols = args['hyper_params'][1:] + cols

        param_values = []
        for hp_name in args['hyper_params'][1:]:
            param_values.append(args[hp_name])
        combinations = list(itertools.product(*param_values))

        for data_name in args['dataset']:
            sheets[data_name] = book.add_sheet(data_name, cell_overwrite_ok=True)
            sheets[data_name + '_avg'] = book.add_sheet(data_name + '_avg', cell_overwrite_ok=True)

            # Write Header names
            row0 = sheets[data_name].row(0)
            row_a0 = sheets[data_name+'_avg'].row(0)
            col_id = -1
            for header in cols:
                col_id += 1
                row0.write(col_id, header)
                row_a0.write(col_id, header)

            row_id = 0
            row_a_id = 0
            for setting in combinations:
                folder_suffix = ''
                row = sheets[data_name].row(row_id + 1)
                row_a = sheets[data_name+'_avg'].row(row_a_id + 1)


                for name, value in zip(args['hyper_params'][1:], setting):
                    folder_suffix += "_" + str(value)
                #   print(path.join(data_name, args['timestamp'] + folder_suffix))
                if not path.exists(path.join(data_name, args['timestamp'] + folder_suffix)):
                    #row_a_id += 1
                    continue


                for name, value in zip(args['hyper_params'][1:], setting):
                    row.write(cols.index(name), value)
                    row_a.write(cols.index(name), value)

                folder_suffix = path.join(data_name, args['timestamp']+folder_suffix)

                # Get label types
                if 'label_types' in args['hyper_params']:
                    label_types = args['label_types']
                else:
                    label_types = [name for name in listdir(folder_suffix) if path.isdir(path.join(folder_suffix, name))]

                for label_type in label_types:

                    path_prefix = path.join(folder_suffix, label_type)
                    try:
                            results = np.load(path.join(path_prefix, 'results.npy')).item()
                    except:
                        continue


                    row = sheets[data_name].row(row_id + 1)
                    row_a = sheets[data_name+'_avg'].row(row_a_id + 1)
                    row.write(cols.index('labels'), label_type)
                    row_a.write(cols.index('labels'), label_type)

                    percents = np.sort([int(key) for key in results.keys()]).astype(np.str_)
                    folds = np.sort([int(fold) for fold in results[str(percents[0])].keys()]).astype(np.str_)

                    perc_pos = cols.index('percents')
                    fold_pos = cols.index('folds')

                    for pid, percent in enumerate(percents):
                        row = sheets[data_name].row(row_id+1)
                        row.write(perc_pos, int(percent))
                        row_a = sheets[data_name+'_avg'].row(row_a_id + 1)
                        row_a.write(perc_pos, int(percent))

                        mean_metrics = np.zeros([1, (pid + 1) * n_metrics])
                        for fold in folds:
                            row_id += 1
                            row = sheets[data_name].row(row_id)
                            row.write(fold_pos, int(fold))

                            offset = pid*len(cols)
                            order = []
                            for metric in metric_names:
                                order.append(cols.index(metric)-(5+len(args['hyper_params'][1:])))
                                val = float(results[percent][fold][metric])

                                row.write(cols.index(metric), round(val, 5))
                                mean_metrics[0, offset+cols.index(metric)-(5+len(args['hyper_params'][1:]))] += round(val, 5)

                        row_id += 1
                        row_a_id += 1
                        row = sheets[data_name].row(row_id)
                        row_a = sheets[data_name+'_avg'].row(row_a_id)

                        for i, metric in enumerate(metric_names):
                            val = mean_metrics[0, order[i]]
                            row.write(cols.index(metric), val/len(folds))
                            row_a.write(cols.index(metric), val/len(folds))

                        row_id += 1
                        row = sheets[data_name].row(row_id)
                        for i in range(len(cols)):
                            row.write(i, '')


    # Save it with a time-stamp
    book.save(path.join(save_path, args['timestamp']+'.xls'))
    #book.save(path.join(save_path, 'default.xls'))

