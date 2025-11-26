from typing import TypeVar

T = TypeVar('T', float, complex)

def write_out_stencil_dense_3D(itterables2:'int[:,:]', tmp2:'T[:,:,:]', out:'T[:,:]', npts2:'int[:]', col:int, pds:'int[:]'):
    counter0 = 0
    for i0 in range(itterables2[0][0],itterables2[0][1]):
        counter1 = 0
        for i1 in range(itterables2[1][0],itterables2[1][1]):
            counter2 = 0
            for i2 in range(itterables2[2][0],itterables2[2][1]):
                if(tmp2[pds[0]+counter0, pds[1]+counter1, pds[2]+counter2] != 0):
                    row = i2 + i1*npts2[2] + i0*npts2[2]*npts2[1]
                    out[row,col] = tmp2[pds[0]+counter0, pds[1]+counter1, pds[2]+counter2]
                counter2 += 1
            counter1 += 1
        counter0 += 1

       
def write_out_stencil_dense_2D(itterables2:'int[:,:]', tmp2:'T[:,:]', out:'T[:,:]', npts2:'int[:]', col:int, pds:'int[:]'):
    counter0 = 0
    for i0 in range(itterables2[0][0],itterables2[0][1]):
        counter1 = 0
        for i1 in range(itterables2[1][0],itterables2[1][1]):
            if(tmp2[pds[0]+counter0, pds[1]+counter1] != 0):
                row = i1 + i0*npts2[1]
                out[row,col] = tmp2[pds[0]+counter0, pds[1]+counter1]
            counter1 += 1
        counter0 += 1
        

def write_out_stencil_dense_1D(itterables2:'int[:,:]', tmp2:'T[:]', out:'T[:,:]', npts2:'int[:]', col:int, pds:'int[:]'):
    counter0 = 0
    for i0 in range(itterables2[0][0],itterables2[0][1]):
        if(tmp2[pds[0]+counter0] != 0):
            row = i0
            out[row,col] = tmp2[pds[0]+counter0]
        counter0 += 1
        

def write_out_block_dense_3D(itterables2:'int[:,:]', tmp2:'T[:,:,:]', out:'T[:,:]', npts2:'int[:]', col:int, pds:'int[:]', spoint2list:int):
    counter0 = 0
    for i0 in range(itterables2[0][0],itterables2[0][1]):
        counter1 = 0
        for i1 in range(itterables2[1][0],itterables2[1][1]):
            counter2 = 0
            for i2 in range(itterables2[2][0],itterables2[2][1]):
                if(tmp2[pds[0]+counter0, pds[1]+counter1, pds[2]+counter2] != 0):
                    row = i2 + i1*npts2[2] + i0*npts2[2]*npts2[1]
                    out[spoint2list+row,col] = tmp2[pds[0]+counter0, pds[1]+counter1, pds[2]+counter2]
                counter2 += 1
            counter1 += 1
        counter0 += 1

       
def write_out_block_dense_2D(itterables2:'int[:,:]', tmp2:'T[:,:]', out:'T[:,:]', npts2:'int[:]', col:int, pds:'int[:]', spoint2list:int):
    counter0 = 0
    for i0 in range(itterables2[0][0],itterables2[0][1]):
        counter1 = 0
        for i1 in range(itterables2[1][0],itterables2[1][1]):
            if(tmp2[pds[0]+counter0, pds[1]+counter1] != 0):
                row = i1 + i0*npts2[1]
                out[spoint2list+row,col] = tmp2[pds[0]+counter0, pds[1]+counter1]
            counter1 += 1
        counter0 += 1
        

def write_out_block_dense_1D(itterables2:'int[:,:]', tmp2:'T[:]', out:'T[:,:]', npts2:'int[:]', col:int, pds:'int[:]', spoint2list:int):
    counter0 = 0
    for i0 in range(itterables2[0][0],itterables2[0][1]):
        if(tmp2[pds[0]+counter0] != 0):
            row = i0
            out[spoint2list+row,col] = tmp2[pds[0]+counter0]
        counter0 += 1