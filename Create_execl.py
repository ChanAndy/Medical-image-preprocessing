#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : write_to_execl.py
@Author: Andy
@Date  : 2021/6/21
@Desc  :
@Contact : 1369635303@qq.com
"""

import pandas as pd
import os

# create form and form_header
def create_form(excel_file_name, form_header):
    df = pd.DataFrame(columns=form_header)
    df.to_excel(excel_file_name, index=False)


# insert the information into the form
def add_info_to_form(excel_file_name, content):
    df = pd.read_excel(excel_file_name)
    df["name"] = content
    df.to_excel(excel_file_name, index=False)

if __name__ == "__main__":

    infor_list = ["Andy", "Dufresen"]
    infor_list_pd = pd.DataFrame(infor_list)
    save_xlsx = r".\name.xlsx"
    header_name = ["name"]
    # create form
    create_form(save_xlsx, header_name)
    add_info_to_form(save_xlsx, infor_list_pd)
