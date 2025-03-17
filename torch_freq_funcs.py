import torch

'''
understand torch.outer, torch.einsum, and torch.unsqueeze
'''

aa = torch.rand(256)
bb = torch.rand(32)

cc = torch.outer(aa, bb)
dd = torch.einsum('i,j->ij', aa, bb)
ee = aa.unsqueeze(1) * bb

assert torch.allclose(cc, dd)
assert torch.allclose(cc, ee)


'''
understand torch.stack
'''

aaa = torch.tensor([[1,2,3],[4,5,6]])
bbb = torch.tensor([[7,8,9],[10,11,12]])


ccc = torch.stack([aaa, bbb], dim = -1)
ddd = torch.stack([aaa, bbb], dim = 0)
eee = torch.stack([aaa, bbb], dim = 1)
fff = torch.stack([aaa, bbb], dim = 2)


print(ddd.shape, '\n', ddd)
print(eee.shape, '\n', eee)
print(fff.shape, '\n', fff)

aaaa = torch.tensor([[1,2],[3,4],[5,6]])


print(aaa.reshape(-1))

''' # we will get:
torch.Size([2, 2, 3]) 
 tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]])
torch.Size([2, 2, 3]) 
 tensor([[[ 1,  2,  3],
         [ 7,  8,  9]],

        [[ 4,  5,  6],
         [10, 11, 12]]])
torch.Size([2, 3, 2]) 
 tensor([[[ 1,  7],
         [ 2,  8],
         [ 3,  9]],

        [[ 4, 10],
         [ 5, 11],
         [ 6, 12]]])
tensor([1, 2, 3, 4, 5, 6])
'''


