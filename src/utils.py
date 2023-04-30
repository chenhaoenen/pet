# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2023/04/30 15:34
# --------------------------------------------

def stats_time(start, end, step, total_step):
    t = end -start
    return '{:.3f}'.format((t / step * (total_step - step) / 3600))