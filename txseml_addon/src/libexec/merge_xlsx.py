'''
Author: George Zhao
Date: 2022-07-13 18:43:44
LastEditors: George Zhao
LastEditTime: 2022-07-13 20:06:07
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import pandas as pd


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='EP3_pairwisealigner_controler')
    parser.add_argument('--dir', type=str, required=True, help='Path to dir.')
    parser.add_argument('--optimal', action='store_true',
                        help='_o?')
    parser.add_argument('--sub', action='store_true',
                        help='_submodel?')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to out of excel table.')
    args = parser.parse_args()

    writer = pd.ExcelWriter(args.output, engine='openpyxl')
    for type_n in [1, 2, 3, 4, 6]:
        for cvortt in ['CV', 'TT']:
            file_name = f'T{type_n}_{cvortt}' + \
                ('_o' if args.optimal == True else '')
            file_namewithsub = file_name + \
                ('' if args.sub == False else '.sub')
            table_form_path = os.path.join(
                args.dir, file_namewithsub + '.csv')
            if os.path.exists(table_form_path) == False:
                raise RuntimeError(f"Error: {table_form_path}")
            pd.read_csv(table_form_path, index_col=0,
                        header=0).to_excel(writer, file_name)
    writer.save()
