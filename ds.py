import argparse
import os
import time
import datetime
from deepsramp import *

parser = argparse.ArgumentParser(description='Run MultiSRAMP')

parser.add_argument('-f', '--file', required=True, type=str, help='Path of datafile')
parser.add_argument('-o', '--out', required=True, type=str, help='Directory of results')
parser.add_argument('-m', '--mode', default='full', type=str, help='Mode of MultiSRAMP')
parser.add_argument('-l', '--len', default=400, type=int, help='Half sequence length of MultiSRAMP')
parser.add_argument('-t', '--target', type=str, help='Target cell line')

args = parser.parse_args()

pid = os.getpid()
print('pid: ', pid)
print(datetime.datetime.now())

# main
extra = 'random'
i = args.target
if i in ['mature', 'full']: # deprecated for SRAMP1 
    traindf, testdf = utils.load(args.file)
else:
    df = utils.load(args.file)
    ythdf = utils.load('data/GSE/compare_mature_ythdf_test.data')

trainds = df2ds(df, downsample=False, pos_label=f'{i}_train', neg_label=f'{i}_train_neg')
traindl = ds2dl(trainds, drop_last=True, num_workers=8)

if i == 'ythdf':
    testds = df2ds(ythdf, downsample=False, pos_label=f'{i}_test', neg_label=f'{i}_test_neg')
else:
    testds = df2ds(df, downsample=False, pos_label=f'{i}_test', neg_label=f'{i}_test_neg')
testdl = ds2dl(testds, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
utils.setup_seed(42)

model = SRAMP(mode=args.mode, halfseqlen=args.len).to(device)
loss_fn = TriLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)

print(f'Start to train - [{i}]')
start = time.time()
train(model, loss_fn, optimizer, scheduler, traindl, testdl, epochs=31, device=device)
end = time.time()
test_res = [test_loop(testdl, model, device, loss_fn)]
print(end-start)

torch.save(model.state_dict(), f'{args.out}/{args.mode}_{args.len}_{i}_single_{extra}.model')
utils.save(test_res, f'{args.out}/{args.mode}_{args.len}_{i}_single_{extra}.metrics')


# DP
dps = []
test_res = []
for dpmode in ['onehot', 'enac', 'embedding', 'ensemble']:
    print(dpmode)
    traindl = ds2dl(trainds, drop_last=True, num_workers=8)
    testdl = ds2dl(testds, shuffle=False, num_workers=4)
    if dpmode == 'ensemble':
        dp = DeepPromiseEnsemble(*dps).to(device)
    else:
        dp = DeepPromise(dpmode).to(device)
    loss_fn = DPLoss()
    optimizer = torch.optim.Adam(dp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)
    print(f'Start to train - [{dpmode}]')
    train(dp, loss_fn, optimizer, scheduler, traindl, testdl, epochs=31, device=device)
    test_res += [dp_test_loop(testdl, dp, device, loss_fn)]
    dps += [dp]


torch.save([i.state_dict() for i in dps], f'{args.out}/{args.mode}_{args.len}_{i}_dp_{extra}.model')
utils.save(test_res, f'{args.out}/{args.mode}_{args.len}_{i}_dp_{extra}.metrics')








