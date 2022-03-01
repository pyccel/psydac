# coding: utf-8

import time
def time_count(t_stamp=None, msg=None):
    new_t_stamp = time.time()
    if msg is None:
        msg = ''
    else:
        msg = '['+msg+']'
    if t_stamp:
        print('time elapsed '+msg+': '+repr(new_t_stamp - t_stamp))
    elif len(msg) > 0:
        print('time stamp set for '+msg)
    return new_t_stamp
